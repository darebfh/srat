import streamlit as st
import json
import os
from pathlib import Path
import pandas as pd
from datetime import datetime
from datasets import load_from_disk
import re
from typing import List, Dict, Any
import html

def load_translations_directory():
    """Load all available translation directories."""
    translations_dir = Path("data/translations")
    if not translations_dir.exists():
        return []
    
    # Get all directories that contain translations
    translation_dirs = [
        d for d in translations_dir.iterdir() 
        if d.is_dir() and (d / "translations.json").exists()
    ]
    
    return sorted(translation_dirs, key=lambda x: x.name, reverse=True)

def find_concept_positions(text: str, concepts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Find positions of concepts in the text and return them with metadata.
    
    Args:
        text: The source text
        concepts: List of concept dictionaries with 'term', 'normalized_text', 'confidence_score', etc.
        
    Returns:
        List of concept dictionaries with added 'start_pos' and 'end_pos' fields
    """
    concept_positions = []
    
    for concept in concepts:
        term = concept.get('term', '')
        if not term:
            continue
        
        # Find all occurrences of the term (case-insensitive) in the original text
        pattern = re.escape(term)
        for match in re.finditer(pattern, text, re.IGNORECASE):
            concept_with_pos = concept.copy()
            concept_with_pos['start_pos'] = match.start()
            concept_with_pos['end_pos'] = match.end()
            concept_positions.append(concept_with_pos)
    
    # Sort by position and remove overlapping concepts (keep first occurrence)
    concept_positions.sort(key=lambda x: (x['start_pos'], x['end_pos']))
    
    # Remove overlapping concepts
    filtered_positions = []
    used_positions = set()
    
    for concept in concept_positions:
        start = concept['start_pos']
        end = concept['end_pos']
        
        # Check if this position overlaps with any already used position
        overlap = False
        for used_start, used_end in used_positions:
            if not (end <= used_start or start >= used_end):
                overlap = True
                break
        
        if not overlap:
            filtered_positions.append(concept)
            used_positions.add((start, end))
    
    return filtered_positions

def highlight_concepts_in_text(text: str, concepts: List[Dict[str, Any]]) -> str:
    """
    Create HTML with highlighted concepts showing term, normalized text, and confidence score.
    
    Args:
        text: The source text
        concepts: List of concept dictionaries with position information
        
    Returns:
        HTML string with highlighted concepts
    """
    if not concepts:
        text_escaped = html.escape(text)
        return f'<div style="white-space: pre-wrap; font-family: monospace; padding: 10px; background-color: #f8f9fa; border-radius: 5px;">{text_escaped}</div>'
    
    # Sort concepts by position (reverse order to avoid index shifting)
    concepts_sorted = sorted(concepts, key=lambda x: x['start_pos'], reverse=True)
    
    # Build HTML with highlights
    html_text = text
    
    for concept in concepts_sorted:
        start = concept['start_pos']
        end = concept['end_pos']
        term = concept.get('term', '')
        normalized = concept.get('normalized_text', term)
        confidence = concept.get('confidence_score', 0.0)
        category = concept.get('category', 'Unknown')
        
        # Get the actual text from the original (preserving case)
        actual_term = text[start:end]
        
        # Escape HTML in the term and metadata
        actual_term_escaped = html.escape(actual_term)
        normalized_escaped = html.escape(str(normalized))
        category_escaped = html.escape(str(category))
        
        # Create color based on confidence score
        # Green for high confidence (>0.8), yellow for medium (0.5-0.8), orange for low (<0.5)
        if confidence > 0.8:
            bg_color = "#90EE90"  # Light green
            border_color = "#228B22"  # Forest green
        elif confidence > 0.5:
            bg_color = "#FFE4B5"  # Moccasin
            border_color = "#FF8C00"  # Dark orange
        else:
            bg_color = "#FFB6C1"  # Light pink
            border_color = "#DC143C"  # Crimson
        
        # Create tooltip with information
        tooltip_text = f"Normalized: {normalized_escaped}<br>Confidence: {confidence:.2f}<br>Category: {category_escaped}"
        
        # Create highlighted span
        highlight_html = f'''<span 
            style="background-color: {bg_color}; 
                   border-bottom: 2px solid {border_color}; 
                   padding: 2px 4px; 
                   border-radius: 3px; 
                   cursor: help;
                   position: relative;"
            title="{tooltip_text}"
            data-normalized="{normalized_escaped}"
            data-confidence="{confidence:.2f}"
            data-category="{category_escaped}">
            {actual_term_escaped}
        </span>'''
        
        html_text = html_text[:start] + highlight_html + html_text[end:]
    
    return f'''<div style="white-space: pre-wrap; font-family: monospace; padding: 15px; background-color: #f8f9fa; border-radius: 5px; line-height: 1.6;">{html_text}</div>'''

def main():
    st.title("Medical Report Translation Viewer")
    
    # Sidebar for dataset selection
    st.sidebar.header("Dataset Selection")
    
    # Try to load the raw dataset (Arrow format)
    try:
        dataset = load_from_disk('data/processed/raw_dataset')
        st.sidebar.success("Raw dataset loaded successfully!")
    except Exception as e:
        st.sidebar.error(f"Error loading raw dataset: {str(e)}")
        dataset = None
    
    # Load available translation directories
    translation_dirs = load_translations_directory()
    
    # Main content area
    tab1, tab2, tab3 = st.tabs(["Raw Dataset", "Translations", "Concept Assessment"])
    
    with tab1:
        if dataset is not None:
            st.header("Raw Dataset Explorer")
            
            # Dataset statistics
            st.subheader("Dataset Statistics")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Reports", len(dataset))
            with col2:
                st.metric("Total Patients", len(set(dataset['patient_id'])))
            with col3:
                total_size_mb = sum(dataset['file_size']) / (1024 * 1024)
                st.metric("Total Size", f"{total_size_mb:.2f} MB")
            
            # Partition distribution
            st.subheader("Distribution by Partition")
            partition_counts = pd.Series({
                row['partition']: 1 for row in dataset
            }).value_counts()
            st.bar_chart(partition_counts)
            
            # Report viewer
            st.subheader("Report Viewer")
            
            # Filter options
            col1, col2 = st.columns(2)
            with col1:
                selected_partition = st.selectbox(
                    "Select Partition",
                    options=['All'] + sorted(set(dataset['partition']))
                )
            
            with col2:
                patient_ids = sorted(set(dataset['patient_id']))
                selected_patient = st.selectbox(
                    "Select Patient ID",
                    options=['All'] + patient_ids
                )
            
            # Filter dataset
            filtered_dataset = dataset
            if selected_partition != 'All':
                filtered_dataset = filtered_dataset.filter(
                    lambda x: x['partition'] == selected_partition
                )
            if selected_patient != 'All':
                filtered_dataset = filtered_dataset.filter(
                    lambda x: x['patient_id'] == selected_patient
                )
            
            # Show filtered reports
            if len(filtered_dataset) > 0:
                report_index = st.selectbox(
                    "Select Report",
                    options=range(len(filtered_dataset)),
                    format_func=lambda i: f"{filtered_dataset[i]['study_id']} ({filtered_dataset[i]['file_path']})"
                )
                
                report = filtered_dataset[report_index]
                st.text_area(
                    "Report Content",
                    report['text'],
                    height=300
                )
                
                # Show metadata
                st.json({
                    'patient_id': report['patient_id'],
                    'study_id': report['study_id'],
                    'file_size': f"{report['file_size'] / 1024:.2f} KB",
                    'file_path': report['file_path']
                })
            else:
                st.warning("No reports found with selected filters")
    
    with tab2:
        st.header("Translations Explorer")
        
        if not translation_dirs:
            st.warning("No translations found. Run some translations first!")
        else:
            # Select translation batch
            selected_dir = st.selectbox(
                "Select Translation Batch",
                options=translation_dirs,
                format_func=lambda x: f"Batch {x.name}"
            )
            
            try:
                # Load translation metadata
                with open(selected_dir / "metadata.json", "r") as f:
                    metadata = json.load(f)
                
                # Display metadata
                st.subheader("Translation Metadata")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Source Language", metadata['source_lang'])
                with col2:
                    st.metric("Target Language", metadata['target_lang'])
                with col3:
                    st.metric("Translation Mode", metadata['translation_mode'])
                
                # Load translations
                with open(selected_dir / "translations.json", 'r', encoding='utf-8') as f:
                    translations = json.load(f)
                
                # Add Raw Dataset Viewer in Expander
                with st.expander("View Complete Translation Dataset"):
                    # Convert to pandas DataFrame
                    translations_df = pd.DataFrame(translations)
                    
                    # Add filter options
                    st.subheader("Filter Options")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Filter by model
                        if 'model_used' in translations_df.columns:
                            models = ['All'] + list(translations_df['model_used'].unique())
                            selected_model = st.selectbox("Filter by Model", models)
                            if selected_model != 'All':
                                translations_df = translations_df[translations_df['model_used'] == selected_model]
                    
                    with col2:
                        # Filter by translation mode
                        if 'translation_mode' in translations_df.columns:
                            modes = ['All'] + list(translations_df['translation_mode'].unique())
                            selected_mode = st.selectbox("Filter by Translation Mode", modes)
                            if selected_mode != 'All':
                                translations_df = translations_df[translations_df['translation_mode'] == selected_mode]
                    
                    # Column selection
                    available_columns = translations_df.columns.tolist()
                    default_columns = [
                        'file_path', 'patient_id', 'study_id', 
                        'original_text', 'translated_text',
                        'model_used', 'translation_mode'
                    ]
                    selected_columns = st.multiselect(
                        "Select Columns to Display",
                        available_columns,
                        default=[col for col in default_columns if col in available_columns]
                    )
                    
                    # Display options
                    display_options = st.radio(
                        "Display Format",
                        ["Table", "JSON"],
                        horizontal=True
                    )
                    
                    # Show the data
                    if display_options == "Table":
                        # Add search functionality
                        search_term = st.text_input("Search in dataset", "")
                        if search_term:
                            mask = translations_df[selected_columns].astype(str).apply(
                                lambda x: x.str.contains(search_term, case=False)
                            ).any(axis=1)
                            filtered_df = translations_df[mask]
                        else:
                            filtered_df = translations_df
                        
                        # Display the dataframe with selected columns
                        st.dataframe(
                            filtered_df[selected_columns],
                            use_container_width=True,
                            height=400
                        )
                        
                        # Show summary statistics
                        st.subheader("Summary Statistics")
                        numeric_cols = filtered_df.select_dtypes(include=['int64', 'float64']).columns
                        if not numeric_cols.empty:
                            st.dataframe(filtered_df[numeric_cols].describe())
                    else:
                        # Display as JSON
                        st.json(translations_df[selected_columns].to_dict(orient='records'))
                    
                    # Export options
                    st.subheader("Export Options")
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("Export to CSV"):
                            csv = translations_df[selected_columns].to_csv(index=False)
                            st.download_button(
                                label="Download CSV",
                                data=csv,
                                file_name=f"translations_{selected_dir.name}.csv",
                                mime="text/csv"
                            )
                    with col2:
                        if st.button("Export to JSON"):
                            json_str = translations_df[selected_columns].to_json(orient='records')
                            st.download_button(
                                label="Download JSON",
                                data=json_str,
                                file_name=f"translations_{selected_dir.name}.json",
                                mime="application/json"
                            )
                
                # Continue with the existing translation viewer code...
                st.subheader("Translation Viewer")
                
                if len(translations) > 0:
                    translation_index = st.selectbox(
                        "Select Translation",
                        options=range(len(translations)),
                        format_func=lambda i: f"{translations[i]['study_id']} ({translations[i]['file_path']})"
                    )
                    
                    translation = translations[translation_index]
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.text_area(
                            "Original Text",
                            translation['original_text'],
                            height=300
                        )
                    with col2:
                        st.text_area(
                            "Translated Text",
                            translation['translated_text'],
                            height=300
                        )
                    
                    # Show extracted concepts if available
                    if 'extracted_concepts' in translation:
                        st.subheader("Extracted Medical Concepts")
                        concepts_df = pd.DataFrame(translation['extracted_concepts'])
                        st.dataframe(concepts_df)
                    
                    # Show translation metadata
                    st.subheader("Translation Details")
                    st.json({
                        'model_used': translation.get('model_used', 'N/A'),
                        'num_extracted_concepts': translation.get('num_extracted_concepts', 0),
                        'num_snomed_translations': translation.get('num_snomed_translations', 0)
                    })
                
            except Exception as e:
                st.error(f"Error loading translations: {str(e)}")
    
    with tab3:
        st.header("Concept Assessment")
        
        # Hardcoded path to translations file
        translations_path = Path("data/results/00_gpt-oss:120b_nodict_baseprompt/20250928_092100/translations.json")
        
        try:
            if not translations_path.exists():
                st.error(f"Translations file not found at: {translations_path}")
            else:
                # Load all translations
                with open(translations_path, 'r', encoding='utf-8') as f:
                    all_translations = json.load(f)
                
                # Get unique reports by study_id (preferring entries with extracted_concepts)
                seen_study_ids = {}
                for translation in all_translations:
                    study_id = translation.get('study_id', 'Unknown')
                    
                    # If we haven't seen this study_id yet, add it
                    if study_id not in seen_study_ids:
                        seen_study_ids[study_id] = translation
                    else:
                        # If we've seen it, prefer the one with extracted_concepts
                        existing = seen_study_ids[study_id]
                        existing_has_concepts = 'extracted_concepts' in existing and existing.get('extracted_concepts') and len(existing.get('extracted_concepts', [])) > 0
                        current_has_concepts = 'extracted_concepts' in translation and translation.get('extracted_concepts') and len(translation.get('extracted_concepts', [])) > 0
                        
                        # Replace if current has concepts and existing doesn't
                        if current_has_concepts and not existing_has_concepts:
                            seen_study_ids[study_id] = translation
                
                # Take first 30 unique reports
                unique_translations = list(seen_study_ids.values())[:30]
                translations = unique_translations
                
                st.info(f"Loaded {len(translations)} unique reports (from {len(seen_study_ids)} total unique reports, {len(all_translations)} total entries)")
                
                if len(translations) > 0:
                    # Filter translations that have extracted concepts
                    translations_with_concepts = [
                        t for t in translations 
                        if 'extracted_concepts' in t and t.get('extracted_concepts') and len(t.get('extracted_concepts', [])) > 0
                    ]
                    
                    if not translations_with_concepts:
                        st.warning("No translations with extracted concepts found in this batch.")
                    else:
                        # Select translation
                        translation_index = st.selectbox(
                            "Select Translation",
                            options=range(len(translations_with_concepts)),
                            format_func=lambda i: f"{translations_with_concepts[i]['study_id']} ({translations_with_concepts[i]['file_path']})"
                        )
                        
                        translation = translations_with_concepts[translation_index]
                        original_text = translation.get('original_text', '')
                        concepts = translation.get('extracted_concepts', [])
                        
                        if not concepts:
                            st.warning("No extracted concepts found for this translation.")
                        else:
                            # Find concept positions in text
                            concept_positions = find_concept_positions(original_text, concepts)
                            
                            # Display statistics
                            st.subheader("Concept Statistics")
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("Total Concepts", len(concepts))
                            with col2:
                                st.metric("Unique Concepts", len(set(c.get('term', '') for c in concepts)))
                            with col3:
                                avg_confidence = sum(c.get('confidence_score', 0) for c in concepts) / len(concepts) if concepts else 0
                                st.metric("Avg Confidence", f"{avg_confidence:.2f}")
                            with col4:
                                found_in_text = len(concept_positions)
                                st.metric("Found in Text", found_in_text)
                            
                            # Filter options
                            st.subheader("Filter Options")
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                min_confidence = st.slider(
                                    "Minimum Confidence Score",
                                    min_value=0.0,
                                    max_value=1.0,
                                    value=0.0,
                                    step=0.05
                                )
                            
                            with col2:
                                categories = sorted(set(c.get('category', 'Unknown') for c in concepts))
                                selected_categories = st.multiselect(
                                    "Filter by Category",
                                    options=categories,
                                    default=categories
                                )
                            
                            # Filter concepts
                            filtered_concepts = [
                                c for c in concepts
                                if c.get('confidence_score', 0) >= min_confidence
                                and c.get('category', 'Unknown') in selected_categories
                            ]
                            
                            # Recalculate positions with filtered concepts
                            filtered_positions = find_concept_positions(original_text, filtered_concepts)
                            
                            # Display highlighted text
                            st.subheader("Source Text with Highlighted Concepts")
                            st.markdown(
                                """
                                <style>
                                .concept-legend {
                                    display: flex;
                                    gap: 15px;
                                    margin-bottom: 10px;
                                    padding: 10px;
                                    background-color: #f0f0f0;
                                    border-radius: 5px;
                                }
                                .legend-item {
                                    display: flex;
                                    align-items: center;
                                    gap: 5px;
                                }
                                .legend-color {
                                    width: 20px;
                                    height: 20px;
                                    border-radius: 3px;
                                    border: 2px solid;
                                }
                                </style>
                                """,
                                unsafe_allow_html=True
                            )
                            
                            # Legend
                            st.markdown(
                                """
                                <div class="concept-legend">
                                    <div class="legend-item">
                                        <div class="legend-color" style="background-color: #90EE90; border-color: #228B22;"></div>
                                        <span>High Confidence (>0.8)</span>
                                    </div>
                                    <div class="legend-item">
                                        <div class="legend-color" style="background-color: #FFE4B5; border-color: #FF8C00;"></div>
                                        <span>Medium Confidence (0.5-0.8)</span>
                                    </div>
                                    <div class="legend-item">
                                        <div class="legend-color" style="background-color: #FFB6C1; border-color: #DC143C;"></div>
                                        <span>Low Confidence (<0.5)</span>
                                    </div>
                                </div>
                                """,
                                unsafe_allow_html=True
                            )
                            
                            # Show highlighted text
                            highlighted_html = highlight_concepts_in_text(original_text, filtered_positions)
                            st.markdown(highlighted_html, unsafe_allow_html=True)
                            
                            # Instructions
                            st.info("💡 Hover over highlighted concepts to see normalized text, confidence score, and category.")
                            
                            # Display concepts table
                            st.subheader("Extracted Concepts Details")
                            
                            # Create DataFrame with filtered concepts
                            concepts_df = pd.DataFrame(filtered_concepts)
                            
                            # Reorder columns for better display
                            if not concepts_df.empty:
                                column_order = ['term', 'normalized_text', 'category', 'confidence_score']
                                if 'codes' in concepts_df.columns:
                                    column_order.append('codes')
                                
                                # Select only columns that exist
                                display_columns = [col for col in column_order if col in concepts_df.columns]
                                concepts_df = concepts_df[display_columns]
                                
                                # Format confidence score for display
                                if 'confidence_score' in concepts_df.columns:
                                    concepts_df['confidence_score'] = concepts_df['confidence_score'].apply(lambda x: f"{x:.3f}")
                                
                                st.dataframe(
                                    concepts_df,
                                    use_container_width=True,
                                    height=400
                                )
                                
                                # Download option
                                csv = concepts_df.to_csv(index=False)
                                st.download_button(
                                    label="Download Concepts as CSV",
                                    data=csv,
                                    file_name=f"concepts_{translation.get('study_id', 'unknown')}.csv",
                                    mime="text/csv"
                                )
                            else:
                                st.info("No concepts match the selected filters.")
                else:
                    st.warning("No translations found in the file.")
                    
        except Exception as e:
            st.error(f"Error loading translations: {str(e)}")
            import traceback
            st.code(traceback.format_exc())

if __name__ == "__main__":
    main()
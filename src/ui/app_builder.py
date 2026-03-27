from __future__ import annotations

import gradio as gr

from src.ui.constraints import list_available_pdfs
from src.ui.layout_compose_handlers import (
    continue_layout_compose_to_draft,
    move_layout_compose_block_down,
    move_layout_compose_block_up,
    return_to_outline_review_from_layout_compose,
    save_layout_compose_block,
    select_layout_compose_block,
)
from src.ui.review_handlers import (
    approve_extraction_and_plan_outline,
    approve_outline_and_generate_draft,
    approve_webpage,
    find_templates,
    preview_selected_template,
    request_webpage_revision,
    revise_extraction,
    revise_outline,
    run_extraction,
)
from src.ui.updates import _layout_compose_ui_hidden, _review_accordion_updates, _stage_action_updates

APP_CSS = """
#paperalchemy-preview {
  min-height: 78vh;
  border: 1px solid #d9dde7;
  border-radius: 16px;
  overflow: auto;
  background: #ffffff;
  box-shadow: 0 18px 40px rgba(15, 23, 42, 0.08);
}

#paperalchemy-preview img {
  width: 100%;
  height: auto;
  display: block;
}

#paperalchemy-logs textarea {
  font-family: Consolas, "SFMono-Regular", monospace;
}
"""

def build_app() -> gr.Blocks:
    available_pdfs = list_available_pdfs()
    default_pdf = available_pdfs[0] if available_pdfs else None

    with gr.Blocks(title="PaperAlchemy", css=APP_CSS) as demo:
        search_state = gr.State({"user_constraints": {}, "tags_json_path": "", "candidates": []})
        selected_candidate_state = gr.State(None)
        workflow_thread_state = gr.State("")
        current_render_html_state = gr.State("")

        gr.Markdown("# PaperAlchemy")

        with gr.Row():
            with gr.Column(scale=1, min_width=320):
                gr.Markdown("### Control Panel")

                pdf_dropdown = gr.Dropdown(
                    choices=available_pdfs,
                    value=default_pdf,
                    label="Paper PDF",
                    allow_custom_value=True,
                )
                background_color = gr.Radio(
                    choices=["light", "dark"],
                    value="light",
                    label="Background Color",
                )
                density = gr.Radio(
                    choices=["spacious", "compact"],
                    value="spacious",
                    label="Page Density",
                )
                navigation = gr.Radio(
                    choices=["yes", "no"],
                    value="no",
                    label="Navigation",
                )
                layout = gr.Radio(
                    choices=["parallelism", "rotation"],
                    value="parallelism",
                    label="Layout Style",
                )

                find_templates_button = gr.Button("Find Templates")
                candidates_radio = gr.Radio(
                    choices=[],
                    label="Top 5 Candidates",
                    interactive=False,
                )
                step1_button = gr.Button(
                    "Step 1: Extract Source Pack",
                    variant="primary",
                    interactive=False,
                )
                with gr.Accordion("Reader Extraction Review", open=False) as paper_review_accordion:
                    paper_markdown = gr.Markdown(
                        value="Run Step 1 to extract the paper into a reviewable source pack."
                    )
                with gr.Accordion("Planned Webpage Outline", open=False) as outline_review_accordion:
                    outline_markdown = gr.Markdown(
                        value="Approve the Reader extraction to generate a reviewable webpage outline."
                    )
                with gr.Group(visible=False) as stage_actions_group:
                    gr.Markdown("### Current Stage Actions")
                    feedback_text = gr.Textbox(
                        label="Human Feedback",
                        lines=4,
                        value="",
                        visible=False,
                    )
                    manual_layout_compose_checkbox = gr.Checkbox(
                        label="Enable Manual Layout Compose",
                        value=False,
                        visible=False,
                    )
                    feedback_images = gr.File(
                        label="Reference Screenshots (Optional)",
                        file_count="multiple",
                        file_types=["image"],
                        type="filepath",
                        value=None,
                        visible=False,
                    )
                    revise_button = gr.Button(
                        "Revise Extraction",
                        variant="secondary",
                        interactive=False,
                        visible=False,
                    )
                    approve_button = gr.Button(
                        "Approve Extraction & Plan Outline",
                        variant="primary",
                        interactive=False,
                        visible=False,
                    )
                    revise_outline_button = gr.Button(
                        "Revise Outline",
                        variant="secondary",
                        interactive=False,
                        visible=False,
                    )
                    approve_outline_button = gr.Button(
                        "Approve Outline & Generate First Draft",
                        variant="primary",
                        interactive=False,
                        visible=False,
                    )
                    request_revision_button = gr.Button(
                        "Request Webpage Revision",
                        variant="secondary",
                        interactive=False,
                        visible=False,
                    )
                    approve_webpage_button = gr.Button(
                        "Approve Final Webpage",
                        variant="primary",
                        interactive=False,
                        visible=False,
                    )
                system_logs = gr.Textbox(
                    label="System Logs",
                    value=(
                        "Choose constraints, click Find Templates, preview a candidate, then run Step 1 to extract "
                        "a reviewable source pack. Use Human Feedback plus Revise Extraction until satisfied, then "
                        "approve the extraction to plan the webpage outline. Revise the outline until it matches the "
                        "sections you want on the final page, optionally enable Layout Compose before generating the "
                        "first draft, then attach "
                        "screenshots and request webpage revisions through the Translator loop until the draft is ready to approve."
                    ),
                    lines=24,
                    interactive=False,
                    elem_id="paperalchemy-logs",
                )

            with gr.Column(scale=2):
                preview_image = gr.Image(
                    value=None,
                    type="filepath",
                    interactive=False,
                    label="Live Webpage Preview",
                    elem_id="paperalchemy-preview",
                )
                with gr.Row():
                    with gr.Column(scale=1, min_width=280):
                        layout_compose_blocks_markdown = gr.Markdown(
                            value="",
                            visible=False,
                        )
                        layout_compose_block_radio = gr.Radio(
                            choices=[],
                            label="Active Block",
                            interactive=False,
                            visible=False,
                        )
                    with gr.Column(scale=1, min_width=360):
                        layout_compose_template_image = gr.Image(
                            value=None,
                            type="filepath",
                            interactive=False,
                            label="Template Section Map",
                            visible=False,
                        )
                    with gr.Column(scale=1, min_width=360):
                        layout_compose_editor_markdown = gr.Markdown(
                            value="",
                            visible=False,
                        )
                        layout_compose_section_gallery = gr.Gallery(
                            value=[],
                            label="Top 6 Section Crops",
                            visible=False,
                            columns=2,
                            height="auto",
                        )
                        layout_compose_section_radio = gr.Radio(
                            choices=[],
                            label="All Compatible Sections",
                            interactive=False,
                            visible=False,
                        )
                        layout_compose_figure_gallery = gr.Gallery(
                            value=[],
                            label="Related Paper Figures",
                            visible=False,
                            columns=2,
                            height="auto",
                        )
                        layout_compose_figure_checkbox = gr.CheckboxGroup(
                            choices=[],
                            label="Selected Figures",
                            interactive=False,
                            visible=False,
                        )
                        layout_compose_validation_markdown = gr.Markdown(
                            value="",
                            visible=False,
                        )
                        with gr.Row():
                            layout_compose_move_up_button = gr.Button(
                                "Move Up",
                                variant="secondary",
                                interactive=False,
                                visible=False,
                            )
                            layout_compose_move_down_button = gr.Button(
                                "Move Down",
                                variant="secondary",
                                interactive=False,
                                visible=False,
                            )
                        with gr.Row():
                            layout_compose_save_button = gr.Button(
                                "Save Block",
                                variant="secondary",
                                interactive=False,
                                visible=False,
                            )
                            layout_compose_continue_button = gr.Button(
                                "Continue To Draft",
                                variant="primary",
                                interactive=False,
                                visible=False,
                            )
                        layout_compose_return_button = gr.Button(
                            "Return To Outline",
                            variant="secondary",
                            interactive=False,
                            visible=False,
                        )

        stage_action_outputs = [
            stage_actions_group,
            feedback_text,
            feedback_images,
            manual_layout_compose_checkbox,
            revise_button,
            approve_button,
            revise_outline_button,
            approve_outline_button,
            request_revision_button,
            approve_webpage_button,
        ]

        compose_outputs = [
            layout_compose_blocks_markdown,
            layout_compose_block_radio,
            layout_compose_template_image,
            layout_compose_editor_markdown,
            layout_compose_section_gallery,
            layout_compose_section_radio,
            layout_compose_figure_gallery,
            layout_compose_figure_checkbox,
            layout_compose_validation_markdown,
            layout_compose_move_up_button,
            layout_compose_move_down_button,
            layout_compose_save_button,
            layout_compose_continue_button,
            layout_compose_return_button,
        ]

        find_templates_button.click(
            fn=find_templates,
            inputs=[background_color, density, navigation, layout],
            outputs=[
                candidates_radio,
                search_state,
                selected_candidate_state,
                system_logs,
                preview_image,
                step1_button,
                paper_markdown,
                outline_markdown,
                paper_review_accordion,
                outline_review_accordion,
                workflow_thread_state,
                current_render_html_state,
                *stage_action_outputs,
                *compose_outputs,
            ],
            api_name="find_templates",
        )

        candidates_radio.change(
            fn=preview_selected_template,
            inputs=[candidates_radio, search_state, system_logs, preview_image],
            outputs=[
                preview_image,
                selected_candidate_state,
                system_logs,
                step1_button,
                paper_markdown,
                outline_markdown,
                paper_review_accordion,
                outline_review_accordion,
                workflow_thread_state,
                current_render_html_state,
                *stage_action_outputs,
                *compose_outputs,
            ],
            api_name="preview_template",
        )

        step1_button.click(
            fn=run_extraction,
            inputs=[pdf_dropdown, candidates_radio, search_state, selected_candidate_state, system_logs],
            outputs=[
                system_logs,
                paper_markdown,
                outline_markdown,
                paper_review_accordion,
                outline_review_accordion,
                workflow_thread_state,
                current_render_html_state,
                *stage_action_outputs,
                *compose_outputs,
            ],
            api_name="extract_and_review",
        )

        revise_button.click(
            fn=revise_extraction,
            inputs=[feedback_text, feedback_images, workflow_thread_state, system_logs, paper_markdown, outline_markdown],
            outputs=[
                system_logs,
                paper_markdown,
                outline_markdown,
                paper_review_accordion,
                outline_review_accordion,
                *stage_action_outputs,
                *compose_outputs,
            ],
            api_name="revise_extraction",
        )

        approve_button.click(
            fn=approve_extraction_and_plan_outline,
            inputs=[workflow_thread_state, system_logs, paper_markdown, outline_markdown],
            outputs=[
                system_logs,
                paper_markdown,
                outline_markdown,
                paper_review_accordion,
                outline_review_accordion,
                current_render_html_state,
                *stage_action_outputs,
                *compose_outputs,
            ],
            api_name="approve_extraction_and_plan_outline",
        )

        revise_outline_button.click(
            fn=revise_outline,
            inputs=[feedback_text, feedback_images, workflow_thread_state, system_logs, outline_markdown],
            outputs=[
                system_logs,
                outline_markdown,
                paper_review_accordion,
                outline_review_accordion,
                *stage_action_outputs,
                *compose_outputs,
            ],
            api_name="revise_outline",
        )

        approve_outline_button.click(
            fn=approve_outline_and_generate_draft,
            inputs=[
                workflow_thread_state,
                system_logs,
                outline_markdown,
                manual_layout_compose_checkbox,
                preview_image,
                current_render_html_state,
            ],
            outputs=[
                system_logs,
                outline_markdown,
                paper_review_accordion,
                outline_review_accordion,
                preview_image,
                current_render_html_state,
                *stage_action_outputs,
                *compose_outputs,
            ],
            api_name="approve_outline_and_generate_draft",
        )

        layout_compose_block_radio.change(
            fn=select_layout_compose_block,
            inputs=[
                layout_compose_block_radio,
                workflow_thread_state,
                system_logs,
            ],
            outputs=[
                system_logs,
                *compose_outputs,
            ],
            api_name="select_layout_compose_block",
        )

        layout_compose_save_button.click(
            fn=save_layout_compose_block,
            inputs=[
                layout_compose_block_radio,
                layout_compose_section_radio,
                layout_compose_figure_checkbox,
                workflow_thread_state,
                system_logs,
            ],
            outputs=[
                system_logs,
                *compose_outputs,
            ],
            api_name="save_layout_compose_block",
        )

        layout_compose_move_up_button.click(
            fn=move_layout_compose_block_up,
            inputs=[
                layout_compose_block_radio,
                layout_compose_section_radio,
                layout_compose_figure_checkbox,
                workflow_thread_state,
                system_logs,
            ],
            outputs=[
                system_logs,
                *compose_outputs,
            ],
            api_name="move_layout_compose_block_up",
        )

        layout_compose_move_down_button.click(
            fn=move_layout_compose_block_down,
            inputs=[
                layout_compose_block_radio,
                layout_compose_section_radio,
                layout_compose_figure_checkbox,
                workflow_thread_state,
                system_logs,
            ],
            outputs=[
                system_logs,
                *compose_outputs,
            ],
            api_name="move_layout_compose_block_down",
        )

        layout_compose_continue_button.click(
            fn=continue_layout_compose_to_draft,
            inputs=[
                layout_compose_block_radio,
                layout_compose_section_radio,
                layout_compose_figure_checkbox,
                workflow_thread_state,
                system_logs,
                outline_markdown,
                preview_image,
                current_render_html_state,
            ],
            outputs=[
                system_logs,
                outline_markdown,
                paper_review_accordion,
                outline_review_accordion,
                preview_image,
                current_render_html_state,
                *stage_action_outputs,
                *compose_outputs,
            ],
            api_name="continue_layout_compose_to_draft",
        )

        layout_compose_return_button.click(
            fn=return_to_outline_review_from_layout_compose,
            inputs=[
                workflow_thread_state,
                system_logs,
                outline_markdown,
                preview_image,
                current_render_html_state,
            ],
            outputs=[
                system_logs,
                outline_markdown,
                paper_review_accordion,
                outline_review_accordion,
                preview_image,
                current_render_html_state,
                *stage_action_outputs,
                *compose_outputs,
            ],
            api_name="return_to_outline_review_from_layout_compose",
        )

        request_revision_button.click(
            fn=request_webpage_revision,
            inputs=[feedback_text, feedback_images, workflow_thread_state, system_logs, preview_image, current_render_html_state],
            outputs=[
                system_logs,
                paper_review_accordion,
                outline_review_accordion,
                preview_image,
                current_render_html_state,
                *stage_action_outputs,
                *compose_outputs,
            ],
            api_name="request_webpage_revision",
        )

        approve_webpage_button.click(
            fn=approve_webpage,
            inputs=[workflow_thread_state, system_logs, preview_image, current_render_html_state],
            outputs=[
                system_logs,
                paper_review_accordion,
                outline_review_accordion,
                preview_image,
                current_render_html_state,
                *stage_action_outputs,
                *compose_outputs,
            ],
            api_name="approve_webpage",
        )

    return demo

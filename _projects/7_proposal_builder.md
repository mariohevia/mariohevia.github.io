---
layout: page
title: Proposal Builder
description: A web-based proposal generation tool with LLM-assisted drafting, human-in-the-loop controls, and a Supabase-backed data layer.
img: assets/img/project_images/proposal_builder.png
importance: 1
category: Professional
featured_home: false
---

Built a web-based proposal builder that combines LLM-assisted drafting with structured inputs, validation, and user review. The system uses an n8n-orchestrated workflow to run multi-step prompt pipelines (with branching logic and error handling), a WeWeb front end for data entry and editing, and a Supabase backend to store proposals, run metadata, and user feedback under access-controlled, privacy-conscious policies.

<div class="row">
    <div style="width: 70%; margin: 0 auto;">
    {% include figure.html
       path="assets/img/project_images/proposal_builder.png"
       title="Proposal Builder"
       class="img-fluid rounded z-depth-1"
    %}
</div>
</div>
<div class="caption">
    Proposal Builder interface showing the split editing view, with a rich text proposal editor (blured) on the left <br>and an AI-assisted chat panel on the right for requesting targeted revisions and refinements.
</div>

### **Project description**

The Proposal Builder is a guided, web-based tool designed to help users produce consistent, high-quality client proposals with less manual effort, while keeping the user in control of the final output. Rather than relying on a single “generate” action, the application structures the drafting process into steps, enabling validation, edits, and selective regeneration at each stage.

The front end is implemented in WeWeb to provide a form-led experience for collecting key project information and allowing users to review and refine generated text. Generation and enrichment are handled by an n8n workflow that orchestrates a multi-stage LLM prompt pipeline, including conditional paths, safeguards, and retries where appropriate. Outputs, run logs, and user feedback are stored in Supabase to support traceability, iteration, and quality improvements over time.

Key objectives of the project include:

- Reducing time-to-first-draft while maintaining proposal consistency and clarity
- Enforcing structured inputs and validation to improve generation quality
- Supporting human-in-the-loop editing, regeneration, and review before export/use
- Capturing run metadata and user feedback to enable continuous prompt/workflow improvements
- Using a Supabase-backed data layer with access controls and privacy protections for stored content and feedback

#### Core components

- **WeWeb front end:** structured forms for input capture, proposal preview/editing, and user feedback
- **n8n orchestration:** multi-step workflow execution with validation, branching logic, and error handling
- **LLM prompt pipeline:** staged generation to produce more reliable outputs than a single-pass prompt
- **Supabase backend:** persistence for proposals, run history/metadata, and user feedback with controlled access
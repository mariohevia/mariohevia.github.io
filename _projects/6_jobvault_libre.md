---
layout: page
title: JobVault Libre
description: An open-source desktop application for managing job applications locally and privately.
img: assets/img/project_images/jobvault_libre.png
importance: 0
category: Open Source (Owner)
github: https://github.com/mariohevia/crime_data_uk/
featured_home: true
---


Built a free and open-source desktop application for managing job applications locally and privately, with a modular PyQt6 interface, YAML-driven configuration, and AppImage distribution for Linux. The project is Free Software in the GNU sense and released under a copyleft licence. The source code is available in this [Github repository](https://github.com/mariohevia/JobVault-Libre)

<div class="row">
    <div style="width: 40%; margin: 0 auto;">
    {% include figure.html
       path="assets/img/project_images/jobvault_libre.png"
       title="JobVault Libre"
       class="img-fluid rounded z-depth-1"
    %}
</div>
</div>
<div class="caption">
    Screenshots of the application. Download the latest release in <a href="https://github.com/mariohevia/JobVault-Libre/releases/latest" target="_blank">here</a>.
</div>

### **Project description**

JobVault Libre is a privacy-centred, open-source job application tracker designed for users who want full control over their data. JobVault Libre runs entirely on your own machine and stores all information locally, providing a secure and dependable alternative to cloud-based job-tracker tools.

Job searching often involves keeping track of postings, deadlines, application materials, follow-ups, and outcomes. Many existing tools address this by storing your information on external servers or requiring an online account, often tethered to subscription models. JobVault Libre takes a different approach. It provides a native, local-only desktop experience with no cloud component, no dependency on web services, and no requirement to share any personal information with third parties.

JobVault Libre offers a structured and private environment for managing every part of the job-application process. You can record job details, track application statuses, manage important dates and actions, store notes, and document submitted materials. All data remains on your machine and can be searched instantly. The application is free and open-source (Free Software, copyleft), giving users full transparency and the ability to extend or modify it.

Key objectives of the project include:

- Providing a local-only job tracker that preserves user privacy
- Offering a clean, straightforward, native desktop interface
- Maintaining an open-source codebase that users can audit and extend
- Avoiding cloud storage, online accounts, and telemetry
- Remaining lightweight and suitable for offline-first workflows

JobVault Libre is distributed as a portable AppImage for Linux, ensuring a consistent native experience across desktop environments and avoiding installation overhead. The UI follows system palettes and theme-aware styling, integrating smoothly on major Linux distributions.

JobVault Libre is an early but functional public release and continues to evolve. Current development focuses on usability improvements, configuration robustness, and long-term extensibility.

#### Upcoming Features

- CV builder expansion, including structured CV generation based on stored application data and profile details.
- AI-assisted CV tailoring using lightweight NLP methods (non-LLM) to help select relevant achievements, job summaries, skills, and projects for a specific job posting.
- Browser extension support to capture job information directly from websites and import it into the local database.
- Optional integration with LLM APIs or local LLMs for users who want additional AI support in drafting or refining application materials.
- General extensibility improvements, including more robust configuration overlays, expanded widget modules, and support for additional workflow customisation.

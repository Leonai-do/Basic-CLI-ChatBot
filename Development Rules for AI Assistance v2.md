### **The Definitive Software Development Blueprint & Rules of Engagement**

**Preamble:** This document serves as the single, authoritative source of truth for all software development. It outlines the core architectural principles, cross-cutting concerns, and immutable rules that govern our work. Adherence to this blueprint is mandatory to ensure consistency, quality, scalability, and maintainability across all projects. Deviation is not permitted without an explicit, logged decision in the project's `decisions.md` file.

---

### **Table of Contents**

*   **High-Level Project Structure**
*   **Part 1: Guiding Principles**
    *   1.1. Architectural Principles (The "-ilities")
    *   1.2. Cross-Cutting Concerns
    *   1.3. Operational Excellence & Production Readiness
*   **Part 2: The Immutable Rules of Engagement**
    *   2.1. The Development Environment & Tooling
    *   2.2. The Development Lifecycle
    *   2.3. Code Style & Formatting (Non-Negotiable)
    *   2.4. Agent Interaction Protocol
*   **Part 3: Authoritative Standards Documents**
    *   3.1. Tech Stack
    *   3.2. Code Style Guide
    *   3.3. Development Best Practices
*   **Part 4: Templates**
    *   4.1. The Pull Request Template (`.github/pull_request_template.md`)
*   **Appendix: Reference Implementation Case Study ("Insightify")**
    *   Product Mission: Insightify
    *   Product Roadmap: Insightify
    *   Product Decisions Log: Insightify
    *   Spec Requirements Document: User Authentication Flow
    *   Tests Specification: User Authentication Flow
    *   Spec Tasks: User Authentication Flow

---

### **High-Level Project Structure**

All projects developed under this blueprint MUST adhere to the following directory structure. This standardized layout ensures predictability and consistency across the codebase.

```
/your-project-name
├── .github/                # GitHub-specific files, including CI/CD workflows and templates.
│   ├── workflows/          # GitHub Actions for CI/CD.
│   │   └── main.yml
│   └── pull_request_template.md # The mandatory template for all PRs.
│
├── app/                    # The main application source code (FastAPI backend).
│   ├── api/                # API layer: routers, endpoints, dependencies.
│   ├── core/               # Core business logic (no framework dependencies).
│   ├── db/                 # Database layer: SQLAlchemy models and Alembic migrations.
│   └── schemas/            # Pydantic schemas for data validation.
│
├── src/                    # The main application source code (React frontend).
│   └── components/
│       ├── ui/             # Atoms (e.g., Button.jsx)
│       ├── composed/       # Molecules (e.g., SearchBar.jsx)
│       └── features/       # Organisms (e.g., UserProfileCard.jsx)
│
├── tests/                  # All backend tests.
│   ├── core/
│   └── api/
│
├── .dockerignore           # Specifies files to exclude from Docker builds.
├── .env.example            # Template for required environment variables.
├── .gitignore              # Specifies files to be ignored by Git.
├── docker-compose.yml      # Defines and runs multi-container Docker applications.
├── Dockerfile              # Instructions to build the application's Docker image.
├── package.json            # Frontend project manifest and dependencies.
└── README.md               # Project overview and setup instructions.
```

---

### **Part 1: Guiding Principles**

This section outlines the core design philosophies that guide all architectural and technical decisions.

#### **1.1. Architectural Principles (The "-ilities")**

*   **Modularity & Decoupling:** Components MUST be self-contained (high cohesion) with minimal dependencies on other components (low coupling). **This is why we use Clean Architecture.**
*   **API-First Design:** The API contract is the most critical artifact. It MUST be designed, documented, and approved *before* implementation begins.
*   **Scalability & Performance:** The system MUST be designed to handle growth. This includes using **asynchronous operations** for all I/O, leveraging **strategic caching** (Redis), and ensuring all application services are **stateless**.
*   **Resiliency & Fault Tolerance:** The system MUST anticipate and gracefully handle failures. This includes implementing **timeouts and retries** for external calls, exposing **`/health` check endpoints**, and designing for **graceful degradation** of non-critical services.

#### **1.2. Cross-Cutting Concerns**

*   **Structured Logging:** All log output MUST be in **JSON** format, including a timestamp, level, service name, and relevant context (e.g., `request_id`).
*   **Configuration Management:** Configuration MUST be externalized from the code. It is to be loaded from environment variables using `pydantic-settings`.
*   **Error Handling:** We MUST distinguish between expected client errors (return specific HTTP status codes) and unexpected server errors (catch, log full stack trace, return a generic 500 error).

#### **1.3. Operational Excellence & Production Readiness**

*   **Containerization:** Containers MUST be lightweight (minimal base images), secure (run as non-root), and reproducible (use multi-stage builds and `.dockerignore`).
*   **Database Management:** The database is a critical asset. Use **connection pooling** in production, ensure **automated backups** are active, and manage all schema changes via **Alembic migrations** only.
*   **CI/CD Pipeline:** The path to production MUST be a series of automated quality gates in this order: **Lint -> Test -> Build -> Deploy**. A failure at any stage stops the entire process.
*   **Code Reviews:** Reviews are for logic, architecture, and edge cases, not style (linters handle style). The author MUST clearly explain the change in the PR description. The reviewer MUST provide constructive, actionable feedback.
*   **Documentation:** Every project MUST have a comprehensive `README.md` and maintain a `decisions.md` file for Architectural Decision Records (ADRs).

---

### **Part 2: The Immutable Rules of Engagement**

This section outlines the mandatory, step-by-step processes and standards. These are requirements.

#### **2.1. The Development Environment & Tooling**
*   **Rule:** All commands, scripts, and development activities MUST be executed within the **WSL2 (Ubuntu) environment**. The native Windows terminal shall not be used for development.
*   **Rule:** The standard IDE for all projects is **Cursor**. All formatting and linting tools (`ruff`, `eslint`, `prettier`) MUST be integrated into the IDE to provide real-time feedback.

#### **2.2. The Development Lifecycle**
1.  **Source of Truth:** All work MUST begin with a **GitHub Issue**.
2.  **Branching:** A new branch MUST be created from `main` for each issue, following the format `<type>/<issue-number>-<short-description>`.
3.  **TDD:** Development MUST ALWAYS follow the **Test-Driven Development (TDD)** methodology.
4.  **Architectural Adherence:** Code MUST be placed within the correct architectural layer as defined in this document.
5.  **Local Validation:** The entire test suite MUST pass locally with 100% success before pushing.
6.  **Committing:** Commits MUST follow the **Conventional Commits** specification and reference the GitHub Issue.
7.  **Pull Request:** A PR MUST be opened using the `.github/pull_request_template.md`. The full content of this template is provided in Part 4.
8.  **Merging:** A branch can only be merged after all CI checks have passed and the PR has been approved.

#### **2.3. Code Style & Formatting (Non-Negotiable)**
*   The style rules are detailed with full examples in the **Code Style Guide** section of this document. All engineers are required to know and apply them.

#### **2.4. Agent Interaction Protocol**
*   **Guided Execution:** The AI agent MUST present any command (shell, Docker, etc.) for review before execution.
*   **Explicit Approval:** The agent MUST wait for explicit user approval before executing any command.

---

### **Part 3: Authoritative Standards Documents**

This section contains the full, unabridged content for our core standards. These are the single sources of truth for their respective domains.

#### **3.1. Tech Stack**

> Version: 1.1.0
> Last Updated: 2025-07-25
> Scope: Global tech stack defaults for all projects. This document defines the primary technology choices for new projects.

##### **Core Backend Technologies**

- **Framework:** FastAPI (Latest Stable)
- **Language:** Python 3.12+
- **Database:** PostgreSQL (Latest Stable)
- **ORM:** SQLAlchemy with Alembic for migrations
- **Caching:** Redis
- **Background Tasks:** Celery

##### **Frontend Stack**

- **Framework:** React (Latest Stable) with Vite
- **Language:** JavaScript (ESNext) / TypeScript
- **Package Manager:** npm (v11.x+)
- **Environment:** Node.js (v22.x+ LTS)
- **CSS Framework:** TailwindCSS (Latest Stable)
- **UI Components:** Shadcn

##### **AI & Data Stack**

- **Vector Database:** Milvus
- **Primary LLM:** Google Gemini 2.5 Pro
- **MCP Server for Library Documentation:** Our internal implementation, "Context7".
- **Alternative LLMs:** OpenAI GPT-4.1, Claude Sonnet 4, 
- **Embeddings Model:** Ollama with `snowflake-arctic-embed`
- **Agentic Tooling:** Dify for agent deployment and RAG pipelines
- **MCP Toolkit:** We use the standard Docker MCP toolkit for server deployment.

##### **Infrastructure & Deployment**

- **Primary Cloud Provider:** AWS
- **Alternative Providers:** Oracle OCI, Google Cloud
- **Development Environment:** Windows 11 with WSL2 (Ubuntu 22.04 LTS)
- **Containerization:** Docker Desktop & docker-compose
- **Asset Storage:** Cloudflare R2 (Primary), Server Filesystem (Fallback)
- **CI/CD:** GitHub Actions
- **Tunneling:** ngrok

##### **IDE & Tooling**

- **Primary IDE:** Cursor
- **CLI Assistants:** Gemini CLI, Claude Code CLI
- **Linter (Python):** `ruff`
- **Linter (JavaScript):** `ESLint` with Prettier
- **Testing Framework (Python):** `pytest`
- **Testing Framework (JavaScript):** `vitest`

#### **3.2. Code Style Guide**

> Version: 2.0.0
> Last Updated: 2025-07-25
> Scope: Global code style rules with concrete examples. All generated and human-written code MUST adhere to these standards.

##### **Python (PEP 8 Compliant)**

- **Line Length:** 88 characters.
- **Indentation:** 4 spaces.
- **Naming Conventions:**
  - **Functions/Variables/Modules:** `snake_case`
  - **Classes:** `PascalCase`
  - **Constants:** `UPPER_SNAKE_CASE`
- **Strings:** Default to double quotes (`"`). Use f-strings for all string interpolation.
- **Docstrings:** Mandatory Google-style docstrings for all public modules, classes, and functions.

###### **Python Implementation Example**
```python
"""Module for handling user-related operations."""

import datetime

MAX_LOGIN_ATTEMPTS = 5

class UserProfile:
    """Manages user profile data.

    Attributes:
        username: The user's chosen username.
    """

    def __init__(self, username: str):
        """Initializes the UserProfile."""
        self.username = username
```

##### **JavaScript, React, HTML, CSS**

- **Line Length:** 80 characters.
- **Indentation:** 2 spaces.
- **Naming Conventions:**
  - **Variables/Functions:** `camelCase`
  - **React Components:** `PascalCase`
  - **Constants:** `UPPER_SNAKE_CASE`
- **Strings:** Default to single quotes (`'`). Use template literals (`` ` ``) for all string interpolation.
- **Docstrings:** Mandatory JSDoc comments for all exported functions and components.

###### **React Implementation Example**
```jsx
import React, { useState } from 'react';

const MAX_ITEMS = 10;

/**
 * A component that displays a list of items.
 *
 * @param {object} props - The component props.
 * @param {string[]} props.initialItems - The initial list of items.
 * @returns {JSX.Element} The rendered ItemList component.
 */
function ItemList({ initialItems }) {
  const [items, setItems] = useState(initialItems);
  return (
    <div>
      {/* ... */}
    </div>
  );
}
```

##### **Tailwind CSS**

- **Class Formatting:** Use a multi-line style where classes for each responsive breakpoint are on a new, vertically aligned line.
- **Class Sorting:** All Tailwind classes MUST be automatically sorted using the official Prettier plugin for Tailwind CSS.

###### **Tailwind CSS Example**
```html
<button
  class="flex items-center justify-center rounded-md bg-indigo-600 px-4 py-2 text-sm font-semibold text-white
         shadow-sm transition-colors duration-150
         hover:bg-indigo-500
         focus-visible:outline-indigo-600"
>
  Click Me
</button>
```

#### **3.3. Development Best Practices**

> Version: 2.0.0
> Last Updated: 2025-07-25
> Scope: Global strategic development guidelines with concrete implementation examples.

##### **Third-Party Dependencies**

When adding a new dependency, it MUST be vetted properly.
- **Selection Criteria:** Choose popular and **actively maintained** options.
- **Verification:** Check the library's **GitHub repository** for recent commits, active issue resolution, and **clear documentation**.

##### **Architecture**

- **Backend (FastAPI):** Strictly follow a **Clean Architecture** / Ports and Adapters approach.
- **Frontend (React):** Employ a simplified **Atomic Design** approach (`ui/`, `composed/`, `features/`).
- **State Management:** Use `useState` for local, `useContext` for shared, and **Zustand** for global state.

##### **API Design & Data Interchange (RESTful)**

- **Versioning:** Use URL path versioning (e.g., `/api/v1/`).
- **Response Structures:** All JSON responses MUST follow these structures, including support for **pagination**.
  - **Success (Single Item):** `{ "data": { ... } }`
  - **Success (List/Paginated):** `{ "data": [ ... ], "metadata": { ... } }`
  - **Error:** `{ "error": { "type": "...", "message": "..." } }`

##### **Testing (TDD is Mandatory)**

- **Coverage Target:** Minimum **90%** for core logic and **80%** overall.
- **Required Test Types:** Unit (`pytest`/`vitest`), Integration, and End-to-End tests are all mandatory.

##### **Security & Environment**

- **Secrets:** All secrets MUST be loaded from environment variables.
- **Documentation:** A `.env.example` file is required in the project root.
- **Policies:** Implement Rate Limiting and a strict CORS policy.
- **Auditing:** CI/CD pipelines must include a dependency scan.

##### **Database Migrations**

- **Tool:** Use Alembic for all database schema changes.
- **Process:** Generate migrations using `alembic revision --autogenerate -m "Descriptive message"`.

---

### **Part 4: Templates**

This section contains the full content for required files and templates.

#### **4.1. The Pull Request Template (`.github/pull_request_template.md`)**

This is the full markdown content that MUST be saved in the file located at `.github/pull_request_template.md`.

```markdown
## Description

*A clear and concise description of the "what" and "why" behind this change. What problem does it solve? What is the goal?*

### **Related Issue**

*This PR addresses the following GitHub Issue. This is mandatory.*

**Fixes:** # (issue number)

## Type of Change

*Please check the box that best describes the nature of this change.*

- [ ] **Bug Fix**
- [ ] **New Feature**
- [ ] **Breaking Change**
- [ ] **Documentation**
- [ ] **CI/CD**

## How Has This Been Tested?

*A clear description of the tests that you ran to verify your changes.*

- [ ] **Unit Tests:** New and existing unit tests pass locally with my changes.
- [ ] **Integration Tests:** New and existing integration tests pass locally.
- [ ] **End-to-End Tests:** E2E tests for the affected user flows have been added/updated and pass.

## Final Checklist for the Author

*This is your personal quality gate.*

- [ ] My code follows the style guidelines and architectural rules of this project.
- [ ] I have performed a self-review of my own code.
- [ ] I have made corresponding changes to the documentation (e.g., `README.md`, `decisions.md`).
- [ ] My changes generate no new warnings from the linter.
```

---

### **Appendix: Reference Implementation Case Study ("Insightify")**

The following documents are provided as a **complete, high-quality reference example**. They are NOT templates to be copied. Their purpose is to illustrate what the final output of a project's planning and specification phase should look like. Use them as a benchmark for quality and detail.

---

### **The Definitive Software Development Blueprint & Rules of Engagement**

**Preamble:** This document serves as the single, authoritative source of truth for all software development. It outlines the core architectural principles, cross-cutting concerns, and immutable rules that govern our work. Adherence to this blueprint is mandatory to ensure consistency, quality, scalability, and maintainability across all projects. Deviation is not permitted without an explicit, logged decision in the project's `decisions.md` file.

---

### **High-Level Project Structure**

All projects developed under this blueprint MUST adhere to the following directory structure. This standardized layout ensures predictability and consistency across the codebase.

```
/your-project-name
├── .github/                # GitHub-specific files, including CI/CD workflows and templates.
│   ├── workflows/          # GitHub Actions for CI/CD.
│   │   └── main.yml
│   └── pull_request_template.md # The mandatory template for all PRs.
│
├── app/                    # The main application source code (FastAPI backend).
│   ├── api/                # API layer: routers, endpoints, dependencies.
│   ├── core/               # Core business logic (no framework dependencies).
│   ├── db/                 # Database layer: SQLAlchemy models and Alembic migrations.
│   └── schemas/            # Pydantic schemas for data validation.
│
├── src/                    # The main application source code (React frontend).
│   └── components/
│       ├── ui/             # Atoms (e.g., Button.jsx)
│       ├── composed/       # Molecules (e.g., SearchBar.jsx)
│       └── features/       # Organisms (e.g., UserProfileCard.jsx)
│
├── tests/                  # All backend tests.
│   ├── core/
│   └── api/
│
├── .dockerignore           # Specifies files to exclude from Docker builds.
├── .env.example            # Template for required environment variables.
├── .gitignore              # Specifies files to be ignored by Git.
├── docker-compose.yml      # Defines and runs multi-container Docker applications.
├── Dockerfile              # Instructions to build the application's Docker image.
├── package.json            # Frontend project manifest and dependencies.
└── README.md               # Project overview and setup instructions.
```

---

### **Part 1: Guiding Principles**

This section outlines the core design philosophies that guide all architectural and technical decisions.

#### **1.1. Architectural Principles (The "-ilities")**

*   **Modularity & Decoupling:** Components MUST be self-contained (high cohesion) with minimal dependencies on other components (low coupling). **This is why we use Clean Architecture.**
*   **API-First Design:** The API contract is the most critical artifact. It MUST be designed, documented, and approved *before* implementation begins.
*   **Scalability & Performance:** The system MUST be designed to handle growth. This includes using **asynchronous operations** for all I/O, leveraging **strategic caching** (Redis), and ensuring all application services are **stateless**.
*   **Resiliency & Fault Tolerance:** The system MUST anticipate and gracefully handle failures. This includes implementing **timeouts and retries** for external calls, exposing **`/health` check endpoints**, and designing for **graceful degradation** of non-critical services.

#### **1.2. Cross-Cutting Concerns**

*   **Structured Logging:** All log output MUST be in **JSON** format, including a timestamp, level, service name, and relevant context (e.g., `request_id`).
*   **Configuration Management:** Configuration MUST be externalized from the code. It is to be loaded from environment variables using `pydantic-settings`.
*   **Error Handling:** We MUST distinguish between expected client errors (return specific HTTP status codes) and unexpected server errors (catch, log full stack trace, return a generic 500 error).

#### **1.3. Operational Excellence & Production Readiness**

*   **Containerization:** Containers MUST be lightweight (minimal base images), secure (run as non-root), and reproducible (use multi-stage builds and `.dockerignore`).
*   **Database Management:** The database is a critical asset. Use **connection pooling** in production, ensure **automated backups** are active, and manage all schema changes via **Alembic migrations** only.
*   **CI/CD Pipeline:** The path to production MUST be a series of automated quality gates in this order: **Lint -> Test -> Build -> Deploy**. A failure at any stage stops the entire process.
*   **Code Reviews:** Reviews are for logic, architecture, and edge cases, not style (linters handle style). The author MUST clearly explain the change in the PR description. The reviewer MUST provide constructive, actionable feedback.
*   **Documentation:** Every project MUST have a comprehensive `README.md` and maintain a `decisions.md` file for Architectural Decision Records (ADRs).

---

### **Part 2: The Immutable Rules of Engagement**

This section outlines the mandatory, step-by-step processes and standards. These are requirements.

#### **2.1. The Development Environment & Tooling**
*   **Rule:** All commands, scripts, and development activities MUST be executed within the **WSL2 (Ubuntu) environment**. The native Windows terminal shall not be used for development.
*   **Rule:** The standard IDE for all projects is **Cursor**. All formatting and linting tools (`ruff`, `eslint`, `prettier`) MUST be integrated into the IDE to provide real-time feedback.

#### **2.2. The Development Lifecycle**
1.  **Source of Truth:** All work MUST begin with a **GitHub Issue**.
2.  **Branching:** A new branch MUST be created from `main` for each issue, following the format `<type>/<issue-number>-<short-description>`.
3.  **TDD:** Development MUST ALWAYS follow the **Test-Driven Development (TDD)** methodology.
4.  **Architectural Adherence:** Code MUST be placed within the correct architectural layer as defined in this document.
5.  **Local Validation:** The entire test suite MUST pass locally with 100% success before pushing.
6.  **Committing:** Commits MUST follow the **Conventional Commits** specification and reference the GitHub Issue.
7.  **Pull Request:** A PR MUST be opened using the `.github/pull_request_template.md`. The full content of this template is provided in Part 4.
8.  **Merging:** A branch can only be merged after all CI checks have passed and the PR has been approved.

#### **2.3. Code Style & Formatting (Non-Negotiable)**
*   The style rules are detailed with full examples in the **Code Style Guide** section of this document. All engineers are required to know and apply them.

#### **2.4. Agent Interaction Protocol**
*   **Guided Execution:** The AI agent MUST present any command (shell, Docker, etc.) for review before execution.
*   **Explicit Approval:** The agent MUST wait for explicit user approval before executing any command.

---

### **Part 3: Authoritative Standards Documents**

This section contains the full, unabridged content for our core standards. These are the single sources of truth for their respective domains.

#### **3.1. Tech Stack**

> Version: 1.1.0
> Last Updated: 2025-07-25
> Scope: Global tech stack defaults for all projects. This document defines the primary technology choices for new projects.

##### **Core Backend Technologies**

- **Framework:** FastAPI (Latest Stable)
- **Language:** Python 3.12+
- **Database:** PostgreSQL (Latest Stable)
- **ORM:** SQLAlchemy with Alembic for migrations
- **Caching:** Redis
- **Background Tasks:** Celery

##### **Frontend Stack**

- **Framework:** React (Latest Stable) with Vite
- **Language:** JavaScript (ESNext) / TypeScript
- **Package Manager:** npm (v11.x+)
- **Environment:** Node.js (v22.x+ LTS)
- **CSS Framework:** TailwindCSS (Latest Stable)
- **UI Components:** Shadcn

##### **AI & Data Stack**

- **Vector Database:** Milvus
- **Primary LLM:** Google Gemini 2.5 Pro
- **MCP Server for Library Documentation:** Our internal implementation, "Context7".
- **Alternative LLMs:** OpenAI GPT-4.1, Claude Sonnet 4
- **Embeddings Model:** Ollama with `snowflake-arctic-embed`
- **Agentic Tooling:** Dify for agent deployment and RAG pipelines
- **MCP Toolkit:** We use the standard Docker MCP toolkit for server deployment.

##### **Infrastructure & Deployment**

- **Primary Cloud Provider:** AWS
- **Alternative Providers:** Oracle OCI, Google Cloud
- **Development Environment:** Windows 11 with WSL2 (Ubuntu 22.04 LTS)
- **Containerization:** Docker Desktop & docker-compose
- **Asset Storage:** Cloudflare R2 (Primary), Server Filesystem (Fallback)
- **CI/CD:** GitHub Actions
- **Tunneling:** ngrok

##### **IDE & Tooling**

- **Primary IDE:** Cursor
- **CLI Assistants:** Gemini CLI, Claude Code CLI
- **Linter (Python):** `ruff`
- **Linter (JavaScript):** `ESLint` with Prettier
- **Testing Framework (Python):** `pytest`
- **Testing Framework (JavaScript):** `vitest`

#### **3.2. Code Style Guide**

> Version: 2.0.0
> Last Updated: 2025-07-25
> Scope: Global code style rules with concrete examples. All generated and human-written code MUST adhere to these standards.

##### **Python (PEP 8 Compliant)**

- **Line Length:** 88 characters.
- **Indentation:** 4 spaces.
- **Naming Conventions:**
  - **Functions/Variables/Modules:** `snake_case`
  - **Classes:** `PascalCase`
  - **Constants:** `UPPER_SNAKE_CASE`
- **Strings:** Default to double quotes (`"`). Use f-strings for all string interpolation.
- **Docstrings:** Mandatory Google-style docstrings for all public modules, classes, and functions.

###### **Python Implementation Example**
```python
"""Module for handling user-related operations."""

import datetime

MAX_LOGIN_ATTEMPTS = 5

class UserProfile:
    """Manages user profile data.

    Attributes:
        username: The user's chosen username.
    """

    def __init__(self, username: str):
        """Initializes the UserProfile."""
        self.username = username
```

##### **JavaScript, React, HTML, CSS**

- **Line Length:** 80 characters.
- **Indentation:** 2 spaces.
- **Naming Conventions:**
  - **Variables/Functions:** `camelCase`
  - **React Components:** `PascalCase`
  - **Constants:** `UPPER_SNAKE_CASE`
- **Strings:** Default to single quotes (`'`). Use template literals (`` ` ``) for all string interpolation.
- **Docstrings:** Mandatory JSDoc comments for all exported functions and components.

###### **React Implementation Example**
```jsx
import React, { useState } from 'react';

const MAX_ITEMS = 10;

/**
 * A component that displays a list of items.
 *
 * @param {object} props - The component props.
 * @param {string[]} props.initialItems - The initial list of items.
 * @returns {JSX.Element} The rendered ItemList component.
 */
function ItemList({ initialItems }) {
  const [items, setItems] = useState(initialItems);
  return (
    <div>
      {/* ... */}
    </div>
  );
}
```

##### **Tailwind CSS**

- **Class Formatting:** Use a multi-line style where classes for each responsive breakpoint are on a new, vertically aligned line.
- **Class Sorting:** All Tailwind classes MUST be automatically sorted using the official Prettier plugin for Tailwind CSS.

###### **Tailwind CSS Example**
```html
<button
  class="flex items-center justify-center rounded-md bg-indigo-600 px-4 py-2 text-sm font-semibold text-white
         shadow-sm transition-colors duration-150
         hover:bg-indigo-500
         focus-visible:outline-indigo-600"
>
  Click Me
</button>
```

#### **3.3. Development Best Practices**

> Version: 2.0.0
> Last Updated: 2025-07-25
> Scope: Global strategic development guidelines with concrete implementation examples.

##### **Third-Party Dependencies**

When adding a new dependency, it MUST be vetted properly.
- **Selection Criteria:** Choose popular and **actively maintained** options.
- **Verification:** Check the library's **GitHub repository** for recent commits, active issue resolution, and **clear documentation**.

##### **Architecture**

- **Backend (FastAPI):** Strictly follow a **Clean Architecture** / Ports and Adapters approach.
- **Frontend (React):** Employ a simplified **Atomic Design** approach (`ui/`, `composed/`, `features/`).
- **State Management:** Use `useState` for local, `useContext` for shared, and **Zustand** for global state.

##### **API Design & Data Interchange (RESTful)**

- **Versioning:** Use URL path versioning (e.g., `/api/v1/`).
- **Response Structures:** All JSON responses MUST follow these structures, including support for **pagination**.
  - **Success (Single Item):** `{ "data": { ... } }`
  - **Success (List/Paginated):** `{ "data": [ ... ], "metadata": { ... } }`
  - **Error:** `{ "error": { "type": "...", "message": "..." } }`

##### **Testing (TDD is Mandatory)**

- **Coverage Target:** Minimum **90%** for core logic and **80%** overall.
- **Required Test Types:** Unit (`pytest`/`vitest`), Integration, and End-to-End tests are all mandatory.

##### **Security & Environment**

- **Secrets:** All secrets MUST be loaded from environment variables.
- **Documentation:** A `.env.example` file is required in the project root.
- **Policies:** Implement Rate Limiting and a strict CORS policy.
- **Auditing:** CI/CD pipelines must include a dependency scan.

##### **Database Migrations**

- **Tool:** Use Alembic for all database schema changes.
- **Process:** Generate migrations using `alembic revision --autogenerate -m "Descriptive message"`.

---

### **Part 4: Templates**

This section contains the full content for required files and templates.

#### **4.1. The Pull Request Template (`.github/pull_request_template.md`)**

This is the full markdown content that MUST be saved in the file located at `.github/pull_request_template.md`.

```markdown
## Description

*A clear and concise description of the "what" and "why" behind this change. What problem does it solve? What is the goal?*

### **Related Issue**

*This PR addresses the following GitHub Issue. This is mandatory.*

**Fixes:** # (issue number)

## Type of Change

*Please check the box that best describes the nature of this change.*

- [ ] **Bug Fix**
- [ ] **New Feature**
- [ ] **Breaking Change**
- [ ] **Documentation**
- [ ] **CI/CD**

## How Has This Been Tested?

*A clear description of the tests that you ran to verify your changes.*

- [ ] **Unit Tests:** New and existing unit tests pass locally with my changes.
- [ ] **Integration Tests:** New and existing integration tests pass locally.
- [ ] **End-to-End Tests:** E2E tests for the affected user flows have been added/updated and pass.

## Final Checklist for the Author

*This is your personal quality gate.*

- [ ] My code follows the style guidelines and architectural rules of this project.
- [ ] I have performed a self-review of my own code.
- [ ] I have made corresponding changes to the documentation (e.g., `README.md`, `decisions.md`).
- [ ] My changes generate no new warnings from the linter.
```

---

### **Appendix: Reference Implementation Case Study ("Insightify")**

The following documents are provided as a **complete, high-quality reference example**. They are NOT templates to be copied. Their purpose is to illustrate what the final output of a project's planning and specification phase should look like. Use them as a benchmark for quality and detail.

---

#### **Product Mission: Insightify**

> Last Updated: 2025-07-25
> Version: 1.0.0

##### **Pitch**

Insightify is an AI-powered business intelligence platform that helps data analysts and business leaders get answers from their data instantly using natural language, eliminating the need for complex SQL queries and slow reporting cycles.

##### **Users**

###### **Primary Customers**
- **Data Analysts:** Professionals who need to quickly explore data and build reports for stakeholders.
- **Business Leaders (CXOs, VPs):** Decision-makers who need on-demand access to key metrics without technical assistance.

###### **User Personas**
**Sarah, the Data Analyst** (25-35 years old)
- **Role:** Senior Data Analyst at a mid-sized e-commerce company.
- **Context:** Juggles requests from marketing, sales, and operations. Spends most of her day writing and optimizing SQL queries.
- **Pain Points:** Reporting is slow and reactive; by the time she delivers a report, the business has already moved on. She can't keep up with ad-hoc questions.
- **Goals:** To empower business users with self-service analytics and to spend more time on deep, proactive analysis rather than repetitive reporting.

##### **The Problem**

###### **Data is Siloed and Inaccessible**
Most company data is locked away in databases that require technical expertise to access. This creates a bottleneck where business users must wait hours or days for analysts to provide reports, slowing down decision-making.

**Our Solution:** Insightify connects to all major data sources and provides a single, unified interface where anyone can ask questions in plain English.

##### **Differentiators**

###### **From SQL to Answers**
Unlike traditional BI tools like Tableau or PowerBI that still require a significant technical setup, we provide a true natural language interface. Instead of building a dashboard, users just ask: "What was our customer churn rate last quarter in the EMEA region?"

###### **Proactive AI Insights**
While other tools are passive, our AI actively monitors data streams and pushes critical insights to users, such as "Sales in the US region have dropped 15% week-over-week, which is anomalous."

##### **Key Features**

- **Natural Language Query Engine:** Ask complex questions about your data in plain English.
- **Automated Dashboard Builder:** AI generates interactive dashboards based on your questions.
- **Data Source Connectors:** One-click integration with PostgreSQL, Snowflake, BigQuery, and more.
- **Anomaly Detection Alerts:** Get notified automatically about significant changes in your key metrics.

---

#### **Product Roadmap: Insightify**

> Last Updated: 2025-07-25
> Version: 1.0.0
> Status: Planning

##### **Phase 1: Core Foundation (1-2 Months)**

**Goal:** Establish the core infrastructure, data connection pipeline, and user authentication.
**Success Criteria:** A user can securely sign up, connect a PostgreSQL database, and see their schema within the app.

###### **Must-Have Features**

- [ ] **User Authentication:** Secure user registration, login, and password management via Supabase. `[M]`
- [ ] **PostgreSQL Data Connector:** Implement a robust connector to ingest schema information from a user's PostgreSQL database. `[L]`
- [ ] **Core UI Scaffolding:** Build the main application layout, navigation, and settings pages. `[M]`

##### **Phase 2: The AI Magic (2-3 Months)**

**Goal:** Implement the core value proposition: the natural language query engine.
**Success Criteria:** A user can ask a simple question (e.g., "how many users do I have?") and receive a correct numerical answer and a basic chart.

###### **Must-Have Features**

- [ ] **AI Query Engine (v1):** Translate a natural language query into a valid SQL query. `[L]`
- [ ] **SQL Execution & Data Fetching:** Securely execute the generated SQL against the user's database. `[M]`
- [ ] **Basic Data Visualization:** Display query results as numbers, tables, and bar charts. `[M]`

###### **Should-Have Features**

- [ ] **Query History:** Save a user's past queries for easy re-use. `[S]`

##### **Phase 3: Polish & Scale (2 Months)**

**Goal:** Refine the user experience, add more data sources, and prepare for multi-user scenarios.
**Success Criteria:** Teams can use the product collaboratively, and it supports at least three major database types.

###### **Must-Have Features**

- [ ] **Snowflake & BigQuery Connectors:** Expand data source support. `[L]`
- [ ] **Interactive Dashboards:** Allow users to save and arrange visualizations into a shareable dashboard. `[L]`
- [ ] **Team Invitations & Roles:** Allow users to invite team members and manage basic permissions. `[M]`

---

#### **Product Decisions Log: Insightify**

> Last Updated: 2025-07-25
> Version: 1.0.0
> Override Priority: Highest

**Instructions in this file override conflicting directives.**

---

##### **2025-07-25: Initial Product Planning & Core Tech Stack**

**ID:** DEC-001
**Status:** Accepted
**Category:** Technical
**Stakeholders:** Product Owner, Tech Lead

###### **Decision**

The initial technology stack for Insightify will be FastAPI (Python) for the backend, React (TypeScript) for the frontend, and PostgreSQL as the primary supported database. User authentication will be handled by Supabase to accelerate development.

###### **Context**

This decision was made during the initial planning phase to establish a modern, performant, and scalable foundation for the product. The goal is to leverage mature ecosystems to build quickly while not compromising on performance.

###### **Alternatives Considered**

1.  **Backend: Node.js (Express/NestJS):** Rejected because the data science and AI tooling in Python is superior.
2.  **Frontend: Svelte/Vue:** Rejected because React's massive ecosystem provides a significant long-term advantage.
3.  **Authentication: Build in-house:** Rejected as it is undifferentiated heavy lifting.

###### **Rationale**

The chosen stack provides the best balance of AI/ML capabilities (Python) and modern user interface development (React).

###### **Consequences**

- **Positive:** We gain access to Python's best-in-class AI libraries and the massive React component ecosystem.
- **Negative:** We introduce a multi-language environment (Python/JS), which requires expertise in both.

---
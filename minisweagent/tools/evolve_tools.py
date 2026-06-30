import dataclasses
import os
from datetime import datetime
from typing import List, Dict, Any
from pathlib import Path
import jsonlines

from minisweagent.tools.annotation import tool_schema
from minisweagent.tools.base import ToolManager, ToolManagerEnv


@dataclasses.dataclass
class EvolveToolsManagerConfig(ToolManagerEnv):
    BUG_LOCATION_OUTPUT_DIR: Path
    PATCH_OUTPUT_DIR: Path
    BUG_REPORT_OUTPUT_DIR: Path
    VUL_ID: str
    MODEL_CONFIG: dict


class EvolveToolsManager(ToolManager):

    def __init__(self, env: "EvolveToolsManagerConfig"):
        super().__init__(env)

#     @tool_schema(
#         name="submit_fl_results",
#         description="Submit one or more suspected fault locations in the source code, including file path and line "
#                     "number for each.",
#         parameters={
#             "type": "object",
#             "properties": {
#                 "locations": {
#                     "type": "array",
#                     "items": {
#                         "type": "object",
#                         "properties": {
#                             "file_path": {
#                                 "type": "string",
#                                 "description": "Path to the source file containing the fault, e.g., "
#                                                "'src/UserService.java'"
#                             },
#                             "start_line": {
#                                 "type": "integer",
#                                 "description": "Exact line number where the fault starts"
#                             },
#                             "end_line": {
#                                 "type": "integer",
#                                 "description": "Exact line number where the fault ends"
#                             },
#                             "reason": {
#                                 "type": "string",
#                                 "description": "Brief explanation of the likely cause, e.g., 'null pointer "
#                                                "dereference', 'unvalidated input'"
#                             },
#                             "confidence": {
#                                 "type": "number",
#                                 "minimum": 0.0,
#                                 "maximum": 1.0,
#                                 "description": "Confidence score (0.0 = guessing, 1.0 = certain)"
#                             }
#                         },
#                         "required": ["file_path", "start_line", "end_line", "reason", "confidence"]
#                     },
#                     "minItems": 1,
#                     "description": "List of bug locations to submit"
#                 }
#             },
#             "required": ["locations"]
#         }
#     )
#     def submit_fl_results(self, locations: List[Dict[str, Any]]) -> dict:
#         """
#         Persist one or more submitted fault locations.
#         Each
#         with jsonlines.open(location_file_path, 'a') as f:
#                 f.write(data)
#         except Exception as e:
#             return {'returncode': 1, 'output': f"❌ Failed to submit fault localization results: {str(e)}"}
#         return {'returncode': 0, 'output': 'Successfully submitted fault localization results'}

#     @tool_schema(
#         name="submit_solution_report",
#         description="Submit a comprehensive bug fix report, structured into five clear parts: "
#                     "description, analysis, fix process, result, and reflection.",
#         parameters={
#             "type": "object",
#             "properties": {
#                 "description": {
#                     "type": "string",
#                     "description": "What is the bug? Describe the issue and where it was found.",
#                     "example": "The application crashes when submitting a form with special characters in the username field."
#                 },
#                 "analysis": {
#                     "type": "string",
#                     "description": "Why did this bug occur? Technical root cause analysis.",
#                     "example": "The backend does not sanitize user input before passing it to a raw SQL query, leading to syntax errors on quotes and backslashes."
#                 },
#                 "fix_process": {
#                     "type": "string",
#                     "description": "What steps were taken to fix the bug? For each step, include both the action and the observed outcome.",
#                     "example": (
#                             "Step 1: Analyzed error logs and stack trace.\n"
#                             "→ Identified that the crash occurs in '/api/user/create' when username contains single quotes.\n\n"

#                             "Step 2: Replaced raw SQL string concatenation with PreparedStatement.\n"
#                             "→ Compilation successful; test with 'O'Connor' no longer causes SQL syntax error.\n\n"

#                             "Step 3: Added input validation middleware to reject or sanitize special characters.\n"
#                             "→ Requests with dangerous inputs now return 400 Bad Request with clear message."
#                     )
#                 },
#                 "fix_result": {
#                     "type": "string",
#                     "description": "What was the outcome after applying the fix?",
#                     "example": "Form submission now works correctly with special characters. All regression tests passed. No performance impact observed."
#                 },
#                 "reflection": {
#                     "type": "string",
#                     "description": "Lessons learned and suggestions for preventing similar issues.",
#                     "example": "Input validation should be enforced globally. We should adopt an ORM or at least always use prepared statements. Consider adding SAST scanning in CI/CD."
#                 }
#             },
#             "required": ["description", "analysis", "fix_process", "fix_result", "reflection"]
#         }
#     )
#     def submit_solution_report(self,
#                                description: str,
#                                analysis: str,
#                                fix_process: str,
#                                fix_result: str,
#                                reflection: str) -> dict:
#         """
#         Submit a defect resolution report and save it as a Markdown file
#         for easier reading and archiving.
#         """
#         # Get output directory
#         output_dir = self.env.BUG_REPORT_OUTPUT_DIR / self.env.VUL_ID
#         try:
#             os.makedirs(output_dir, exist_ok=True)
#         except Exception as e:
#             return {'returncode': 1, 'output': f"❌ Failed to create report directory: {str(e)}"}

#         # Use a timestamp to generate a unique filename
#         timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#         safe_title = "solution_report"
#         file_name = f"{timestamp}_{safe_title}.md"
#         file_path = output_dir / file_name

#         # Build Markdown content
#         markdown_content = f"""# Bug Fix Report of {self.env.VUL_ID}

# > **Time**: {datetime.now().isoformat()}

# ---

# ## 📝 Description
# {description}

# ---

# ## 🔍 Analysis
# {analysis}

# ---

# ## 🛠️ Fix Process
# {fix_process}

# ---

# ## ✅ Fix Result
# {fix_result}

# ---

# ## 💡 Reflection & Lessons Learned
# {reflection}

# ---
# 📄 *This report was automatically generated by the AI agent.*
# """

#         # Write file
#         try:
#             with open(file_path, "w", encoding="utf-8") as f:
#                 f.write(markdown_content)
#         except Exception as e:
#             return {'returncode': 1, 'output': f"❌ Failed to save report as Markdown: {str(e)}"}

#         # Return a success message (can be fed back to the LLM)
#         short_desc = description.split(".")[0][:60]
#         return {
#             'returncode': 0,
#             'output': (
#                 f"✅ Solution report saved as Markdown: `{file_name}`\n"
#                 f"   → Path: {file_path}\n"
#                 f"   → Summary: \"{short_desc}...\""
#             )
#         }

#     @tool_schema(
#         name="get_latest_solution_report",
#         description="Retrieve the most recently solution report. ",
#         parameters={}
#     )
#     def get_latest_solution_report(self) -> dict:
#         """
#         Get the latest solution report content for the specified vulnerability ID.
#         Return the raw Markdown so the agent can review previous decisions.
#         """
#         # Determine vul_id
#         vul_id =self.env.VUL_ID

#         # Build target directory path
#         report_dir = self.env.BUG_REPORT_OUTPUT_DIR / vul_id

#         if not os.path.exists(report_dir):
#             return {'returncode': 1, 'output': f"❌ No reports found."}

#         # Get all .md files
#         markdown_files = [f for f in os.listdir(report_dir) if f.endswith(".md")]
#         if not markdown_files:
#             return {'returncode': 1, 'output': f"❌ No reports found."}

#         # Sort by filename timestamp prefix and pick the latest one
#         # Filename format: 20250405_142010_solution_report.md
#         try:
#             sorted_files = sorted(
#                 markdown_files,
#                 key=lambda x: x.split('_')[0] + x.split('_')[1],  # Extract timestamp part for sorting
#                 reverse=True
#             )
#             latest_file = sorted_files[0]
#         except Exception:
#             # Fallback: sort by last modified time
#             sorted_files = sorted(
#                 markdown_files,
#                 key=lambda f: os.path.getmtime(report_dir / f),
#                 reverse=True
#             )
#             latest_file = sorted_files[0]

#         file_path = report_dir / latest_file

#         try:
#             with open(file_path, "r", encoding="utf-8") as f:
#                 content = f.read()
#             return {
#                 'returncode': 0, 'output':
#                 f"📄 Latest solution report:\n\n{content}"}
#         except Exception as e:
#             return {'returncode': 1, 'output': f"❌ Failed to read report file: {str(e)}"}
# Each file path maps to one .jsonl log file, and each record is appended as a single line.
#         """
#         try:
#             output_root = self.env.BUG_LOCATION_OUTPUT_DIR / self.env.VUL_ID
#             os.makedirs(output_root, exist_ok=True)

#             data = {
#                 'locations': locations,
#                 'time': datetime.now().isoformat()
#             }

#             location_file_path = output_root / "FL.jsonl"
            
    @tool_schema(
        name="submit_patch",
        description="Submit the final code changes, this action will end the workflow.",
        parameters={
            "type": "object",
            "properties": {},
            "required": []
        }
    )
    def submit_patch(self, **kwargs) -> dict:
        """
        Persist the patch.

        Note: accepts **kwargs to tolerate extra parameters generated by the model,
        such as patch_file, cmd, input, etc. These parameters are ignored.
        """
        # output_root = self.env.PATCH_OUTPUT_DIR / self.env.VUL_ID
        # try:
        #     os.makedirs(output_root, exist_ok=True)
        # except Exception as e:
        #     return {'returncode': 1, 'output': f"❌ Failed to create patch directory: {str(e)}"}

        # data = {
        #     'patch': patch,
        #     'time': datetime.now().isoformat()
        # }

        # location_file_path = output_root / "patch.jsonl"
        # with jsonlines.open(location_file_path, 'a') as f:
        #     f.write(data)
        return {'returncode': 0, 'output': 'Successfully submitted patch'}
        

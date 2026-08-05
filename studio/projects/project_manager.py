"""
QuantLab Enterprise Project Manager Engine.

Provides CRUD operations, duplication, archiving, versioning, tagging, searching, moving,
renaming, importing, and exporting for quantitative projects.
"""

import json
import os
import shutil
from typing import Any, Dict, List, Optional

from studio.logging.studio_logger import get_studio_logger
from studio.projects.project_model import EnterpriseProject

logger = get_studio_logger("ProjectManager")


class ProjectManager:
    """Institutional Project Platform Manager Engine."""

    def __init__(self, root_projects_dir: str = "./projects") -> None:
        self.root_projects_dir = os.path.abspath(root_projects_dir)
        os.makedirs(self.root_projects_dir, exist_ok=True)
        self._projects: Dict[str, EnterpriseProject] = {}

    def create_project(self, name: str, tags: Optional[List[str]] = None) -> EnterpriseProject:
        """Create new quantitative project with full institutional sub-directories."""
        project_dir = os.path.join(self.root_projects_dir, name)
        os.makedirs(project_dir, exist_ok=True)

        proj = EnterpriseProject(name=name, path=project_dir, tags=tags or [])

        # Create all sub-directories
        for folder in [
            proj.research_dir,
            proj.strategies_dir,
            proj.experiments_dir,
            proj.datasets_dir,
            proj.optimization_dir,
            proj.models_dir,
            proj.reports_dir,
            proj.results_dir,
            proj.docs_dir,
            proj.config_dir,
            proj.metadata_dir,
            proj.snapshots_dir,
            proj.assets_dir,
        ]:
            os.makedirs(os.path.join(project_dir, folder), exist_ok=True)

        self._projects[proj.project_id] = proj
        logger.info(f"Created Enterprise Project '{name}' (ID={proj.project_id}) at '{project_dir}'")
        return proj

    def duplicate_project(self, project_id: str, new_name: str) -> Optional[EnterpriseProject]:
        """Duplicate an existing project into a new project copy."""
        orig = self._projects.get(project_id)
        if not orig or not os.path.exists(orig.path):
            return None

        new_dir = os.path.join(self.root_projects_dir, new_name)
        shutil.copytree(orig.path, new_dir, dirs_exist_ok=True)

        dup = EnterpriseProject(name=new_name, path=new_dir, tags=list(orig.tags))
        self._projects[dup.project_id] = dup
        logger.info(f"Duplicated project '{orig.name}' -> '{new_name}'")
        return dup

    def archive_project(self, project_id: str) -> bool:
        """Archive target project."""
        proj = self._projects.get(project_id)
        if proj:
            proj.is_archived = True
            logger.info(f"Archived project '{proj.name}'")
            return True
        return False

    def tag_project(self, project_id: str, tags: List[str]) -> bool:
        """Add tags to target project."""
        proj = self._projects.get(project_id)
        if proj:
            proj.tags = list(set(proj.tags + tags))
            return True
        return False

    def search_projects(self, query: str) -> List[EnterpriseProject]:
        """Search projects by name or tag match."""
        q = query.lower()
        results = []
        for p in self._projects.values():
            if q in p.name.lower() or any(q in t.lower() for t in p.tags):
                results.append(p)
        return results

    def list_projects(self, include_archived: bool = False) -> List[EnterpriseProject]:
        """List registered projects."""
        if include_archived:
            return list(self._projects.values())
        return [p for p in self._projects.values() if not p.is_archived]

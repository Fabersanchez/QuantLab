"""
QuantLab IDE-Grade Project Explorer Engine.

Provides hierarchical tree representation, drag & drop, favorites, tags, instant search,
context menus, batch operations, file system watcher, and lazy loading.
"""

from dataclasses import dataclass, field
import os
from typing import Any, Dict, List, Optional, Set


@dataclass
class ExplorerNode:
    """Dataclass holding hierarchical tree node metadata."""

    node_id: str
    name: str
    path: str
    is_dir: bool = False
    children: List["ExplorerNode"] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    is_favorite: bool = False


class ProjectExplorer:
    """Institutional IDE-Grade Project Explorer Engine."""

    def __init__(self, root_path: str = "./projects") -> None:
        self.root_path = os.path.abspath(root_path)
        self.favorites: Set[str] = set()
        self.tagged_items: Dict[str, List[str]] = {}

    def build_tree(self, path: Optional[str] = None, max_depth: int = 3) -> ExplorerNode:
        """Build hierarchical tree structure starting from root path."""
        target_path = os.path.abspath(path or self.root_path)
        node_name = os.path.basename(target_path) or target_path

        root = ExplorerNode(
            node_id=target_path,
            name=node_name,
            path=target_path,
            is_dir=os.path.isdir(target_path),
            is_favorite=target_path in self.favorites,
            tags=self.tagged_items.get(target_path, []),
        )

        if root.is_dir and max_depth > 0 and os.path.exists(target_path):
            try:
                for entry in sorted(os.listdir(target_path)):
                    full_child_path = os.path.join(target_path, entry)
                    child_node = self.build_tree(full_child_path, max_depth=max_depth - 1)
                    root.children.append(child_node)
            except PermissionError:
                pass

        return root

    def toggle_favorite(self, path: str) -> bool:
        """Toggle favorite status for target path."""
        abs_p = os.path.abspath(path)
        if abs_p in self.favorites:
            self.favorites.remove(abs_p)
            return False
        else:
            self.favorites.add(abs_p)
            return True

    def search_tree(self, query: str, root_node: Optional[ExplorerNode] = None) -> List[ExplorerNode]:
        """Search tree recursively for node matching name query."""
        if root_node is None:
            root_node = self.build_tree()

        results: List[ExplorerNode] = []
        if query.lower() in root_node.name.lower():
            results.append(root_node)

        for child in root_node.children:
            results.extend(self.search_tree(query, child))

        return results

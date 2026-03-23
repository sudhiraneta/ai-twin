from .notes_connector import NotesConnector
from .singularity_db_connector import SingularityDBConnector
from .browser_connector import BrowserConnector
from .task_connector import TaskConnector
from .body_connector import BodyConnector
from .analytics_connector import AnalyticsConnector
from .photos_connector import PhotosConnector

ALL_CONNECTORS = {
    "apple_notes": NotesConnector,
    "singularity_db": SingularityDBConnector,
    "browser": BrowserConnector,
    "tasks": TaskConnector,
    "body": BodyConnector,
    "analytics": AnalyticsConnector,
    "google_photos": PhotosConnector,
}

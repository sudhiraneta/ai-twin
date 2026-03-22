import sys
from datetime import datetime, timedelta

from config import SINGULARITY_AGENT_DIR
from memory.chunker import Chunk, _ensure_metadata
from .base import BaseConnector


class BodyConnector(BaseConnector):
    """Connector for gym and nutrition data via Singularity's body_tracker."""

    source_name = "body"

    def _import_tracker(self):
        if str(SINGULARITY_AGENT_DIR) not in sys.path:
            sys.path.insert(0, str(SINGULARITY_AGENT_DIR))
        import body_tracker
        return body_tracker

    def fetch(self, since: float | None = None, days_back: int = 14) -> list[Chunk]:
        tracker = self._import_tracker()

        data = tracker.analyse_notes(days=days_back)
        nutrition = tracker.analyse_nutrition(days=days_back)

        chunks = []
        week_start = (datetime.now() - timedelta(days=datetime.now().weekday())).strftime("%Y-%m-%d")

        # Gym summary chunk
        gym_days = [d for d, v in data.items() if v.get("gym")]
        gym_count = len(gym_days)
        gym_lines = [
            f"Gym Tracker: Week of {week_start}",
            f"Sessions: {gym_count}/4 target",
        ]
        if gym_days:
            gym_lines.append(f"Gym days: {', '.join(sorted(gym_days))}")
        if gym_count >= 4:
            gym_lines.append("Hit 4x this week.")
        else:
            gym_lines.append(f"{4 - gym_count} sessions left to hit 4x.")

        chunks.append(Chunk(
            text="\n".join(gym_lines),
            metadata=_ensure_metadata({
                "source": self.source_name,
                "conversation_id": f"body_gym_{week_start}",
                "title": f"Gym week of {week_start}",
                "timestamp": f"{week_start}T00:00:00+00:00",
                "msg_timestamp": f"{week_start}T00:00:00+00:00",
                "role": "user",
                "type": "body_gym",
                "pillar": "BODY",
                "dimension": "wellness",
                "classified": "true",
            }),
        ))

        # Nutrition summary chunk
        scores = nutrition.get("scores", {})
        nut_lines = [
            f"Nutrition: Week of {week_start}",
            f"Overall score: {scores.get('Overall', 0)}/10",
        ]
        if nutrition.get("veggies"):
            nut_lines.append(f"Veggies: {', '.join(nutrition['veggies'])}")
        if nutrition.get("proteins"):
            nut_lines.append(f"Protein: {', '.join(nutrition['proteins'])}")
        if nutrition.get("carbs"):
            nut_lines.append(f"Carbs: {', '.join(nutrition['carbs'])}")
        if nutrition.get("fats"):
            nut_lines.append(f"Fats: {', '.join(nutrition['fats'])}")
        if nutrition.get("cheats"):
            nut_lines.append(f"Cheats: {', '.join(nutrition['cheats'])}")
        nut_lines.append(f"Hydration: {'yes' if nutrition.get('hydrated') else 'not logged'}")

        # Hair protocol
        hair = nutrition.get("hair_meds", {})
        hair_status = [f"{med}: {'yes' if found else 'no'}" for med, found in hair.items()]
        if hair_status:
            nut_lines.append(f"Hair protocol: {', '.join(hair_status)}")

        # Red flags
        flags = nutrition.get("red_flags", [])
        if flags:
            nut_lines.append(f"Red flags: {'; '.join(flags)}")

        chunks.append(Chunk(
            text="\n".join(nut_lines),
            metadata=_ensure_metadata({
                "source": self.source_name,
                "conversation_id": f"body_nutrition_{week_start}",
                "title": f"Nutrition week of {week_start}",
                "timestamp": f"{week_start}T00:00:00+00:00",
                "msg_timestamp": f"{week_start}T00:00:00+00:00",
                "role": "user",
                "type": "body_nutrition",
                "pillar": "BODY",
                "dimension": "nutrition",
                "classified": "true",
            }),
        ))

        return chunks

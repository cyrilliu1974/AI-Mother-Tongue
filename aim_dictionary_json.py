import json
import os
from datetime import datetime

class AIMDictionary:
    def __init__(self, file_path="aim_dictionary.json"):
        self.file_path = file_path
        self.entries = []

        if os.path.exists(self.file_path):
            try:
                with open(self.file_path, "r") as f:
                    # Load existing entries. Ensure they are unique and merge counts/versions
                    loaded_entries = json.load(f)
                    # Use a dictionary to store unique entries with latest version and summed usage_count
                    history_map = {}
                    for entry in loaded_entries:
                        key = (entry["aim_id"], entry["human_label"], entry["context"])
                        if key in history_map:
                            history_map[key]["usage_count"] += entry.get("usage_count", 1) # Handle old entries without usage_count
                            # Keep the latest version if multiple entries with same key exist
                            if entry["version"] > history_map[key]["version"]:
                                history_map[key]["version"] = entry["version"]
                        else:
                            history_map[key] = entry
                            history_map[key]["usage_count"] = entry.get("usage_count", 1) # Ensure usage_count exists
                    self.entries = list(history_map.values())

            except json.JSONDecodeError:
                print(f"Warning: {self.file_path} is corrupted or empty. Starting with empty dictionary.")
                self.entries = []

        self._buffer = [] # Buffer for current run's new entries

    def add_entry(self, aim_id_sequence_str, human_label, context):
        # aim_id_sequence_str is expected to be a string representation of a list, e.g., "[1, 5]"
        
        # First, check and update in the buffer for the current run
        found_in_buffer = False
        for entry in self._buffer:
            if (entry["aim_id"] == aim_id_sequence_str and
                entry["human_label"] == human_label and
                entry["context"] == context):
                entry["usage_count"] += 1
                entry["version"] = datetime.now().isoformat() # Update timestamp on usage
                found_in_buffer = True
                break

        if not found_in_buffer:
            new_entry = {
                "aim_id": aim_id_sequence_str,
                "version": datetime.now().isoformat(),
                "human_label": human_label,
                "context": context,
                "usage_count": 1
            }
            self._buffer.append(new_entry)

    def get_entries(self):
        # Merges historical entries with current buffer entries for viewing, not for saving
        # This needs to be careful about duplicates between history and buffer for viewing
        # For simplicity, just concatenate, but for unique view, a map would be better
        # For saving, `save` handles the unique merge correctly.
        return self.entries + self._buffer # Note: this might contain duplicates if buffer items are also in history

    def get_entries_by_id(self, aim_id_sequence_str):
        all_entries = self.get_entries()
        return [entry for entry in all_entries if entry["aim_id"] == aim_id_sequence_str]

    def save(self):
        # Consolidate entries from history and buffer for saving
        consolidated_map = {}
        for entry in self.entries: # Add historical entries first
            key = (entry["aim_id"], entry["human_label"], entry["context"])
            consolidated_map[key] = entry
        
        for new_entry in self._buffer: # Add/update entries from the current buffer
            key = (new_entry["aim_id"], new_entry["human_label"], new_entry["context"])
            if key in consolidated_map:
                # Update usage_count and version for existing entries
                consolidated_map[key]["usage_count"] += new_entry["usage_count"]
                consolidated_map[key]["version"] = new_entry["version"] # Keep the latest timestamp
            else:
                consolidated_map[key] = new_entry # Add new entry

        final_entries_to_save = list(consolidated_map.values())

        with open(self.file_path, "w") as f:
            json.dump(final_entries_to_save, f, indent=2)
        print(f"[AIMDictionary] Saved {len(final_entries_to_save)} entries to {self.file_path}")
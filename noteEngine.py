from llama_index.core.tools import FunctionTool
import os

note_file = os.path.join("data", "notes.txt")


def save_note(note):

    formatted_note = f"{note['title']} :\n{note['content']}"

    if not os.path.exists(note_file):
        open(note_file, "w")

    with open(note_file, "a") as f:
        f.writelines([formatted_note + "\n"])

    return note['content']


note_engine = FunctionTool.from_defaults(
    fn=save_note,
    name="note_saver",
    description="this tool can save a text based note to a file for the user and return the saved info",
)

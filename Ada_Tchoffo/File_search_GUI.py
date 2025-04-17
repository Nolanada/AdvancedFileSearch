import os
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from pathlib import Path
from docx import Document
import PyPDF2
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# -------- File Readers --------
def extract_text_from_txt(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except:
        return ""

def extract_text_from_pdf(file_path):
    try:
        with open(file_path, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            return " ".join(page.extract_text() or "" for page in reader.pages)
    except:
        return ""

def extract_text_from_docx(file_path):
    try:
        doc = Document(file_path)
        return "\n".join(p.text for p in doc.paragraphs)
    except:
        return ""

# -------- Main App --------
class FileSearchApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Intelligent File Search")
        self.root.geometry("800x550")

        self.folder_path = None
        self.files = []
        self.model = SentenceTransformer('paraphrase-MiniLM-L6-v2')  # Pretrained SentenceTransformer

        # --- UI Layout ---
        self.label = tk.Label(root, text="Choose Search Directory", font=("Arial", 13))
        self.label.pack(pady=10)

        self.select_btn = tk.Button(root, text="Select Folder", command=self.select_folder)
        self.select_btn.pack()

        self.search_mode = tk.StringVar(value="name")
        self.dropdown = ttk.Combobox(root, textvariable=self.search_mode, values=["Search by File Name", "Search by File Content", "Semantic Search (AI)"], state="readonly")
        self.dropdown.pack(pady=10)

        self.search_entry = tk.Entry(root, width=50)
        self.search_entry.pack(pady=5)

        self.search_btn = tk.Button(root, text="Search", command=self.run_search)
        self.search_btn.pack(pady=5)

        self.tree = ttk.Treeview(root, columns=('Name', 'Path'), show='headings')
        self.tree.heading('Name', text='File Name')
        self.tree.heading('Path', text='Full Path')
        self.tree.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

    def select_folder(self):
        folder = filedialog.askdirectory()
        if folder:
            self.folder_path = folder
            messagebox.showinfo("Folder Selected", f"Search folder set to:\n{folder}")

    def run_search(self):
        mode = self.search_mode.get()
        query = self.search_entry.get().strip().lower()

        if not self.folder_path:
            messagebox.showwarning("No Folder", "Please select a folder first.")
            return

        if not query:
            messagebox.showwarning("No Query", "Please enter a search query.")
            return

        self.tree.delete(*self.tree.get_children())  # Clear previous results

        if "Name" in mode:
            self.search_by_name(query)
        elif "Content" in mode:
            self.search_by_content(query)
        else:
            self.semantic_search(query)

    def search_by_name(self, query):
        for root_dir, _, files in os.walk(self.folder_path):
            for file in files:
                if query in file.lower():
                    full_path = os.path.join(root_dir, file)
                    self.tree.insert('', tk.END, values=(file, full_path))

    def search_by_content(self, query):
        supported_exts = {'.txt', '.pdf', '.docx'}

        for root_dir, _, files in os.walk(self.folder_path):
            for file in files:
                ext = Path(file).suffix.lower()
                if ext not in supported_exts:
                    continue

                full_path = os.path.join(root_dir, file)
                content = ""
                if ext == '.txt':
                    content = extract_text_from_txt(full_path)
                elif ext == '.pdf':
                    content = extract_text_from_pdf(full_path)
                elif ext == '.docx':
                    content = extract_text_from_docx(full_path)

                if query in content.lower():
                    self.tree.insert('', tk.END, values=(file, full_path))

    def semantic_search(self, query):
        supported_exts = {'.txt', '.pdf', '.docx'}
        query_embedding = self.model.encode([query])[0]  # Encode the query

        # Iterate through files
        for root_dir, _, files in os.walk(self.folder_path):
            for file in files:
                ext = Path(file).suffix.lower()
                if ext not in supported_exts:
                    continue

                full_path = os.path.join(root_dir, file)
                content = ""
                if ext == '.txt':
                    content = extract_text_from_txt(full_path)
                elif ext == '.pdf':
                    content = extract_text_from_pdf(full_path)
                elif ext == '.docx':
                    content = extract_text_from_docx(full_path)

                file_embedding = self.model.encode([content])[0]  # Embed file content

                # Compare using cosine similarity
                similarity = cosine_similarity([query_embedding], [file_embedding])[0][0]

                # If similarity is high enough, display the result
                if similarity > 0.5:  # You can adjust the threshold for relevance
                    self.tree.insert('', tk.END, values=(file, full_path))

# -------- Run App --------
if __name__ == "__main__":
    root = tk.Tk()
    app = FileSearchApp(root)
    root.mainloop()

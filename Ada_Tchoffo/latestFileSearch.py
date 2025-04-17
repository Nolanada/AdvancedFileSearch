import os
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
from sentence_transformers import SentenceTransformer, util
import PyPDF2
import docx
import re
import joblib

# Load or create model
model_path = "best_model.h5"
if os.path.exists(model_path):
    model = joblib.load(model_path)
else:
    model = SentenceTransformer('all-MiniLM-L6-v2')
    joblib.dump(model, model_path)

# Extract text from PDF or DOCX files
def extract_text_from_file(file_path):
    try:
        if file_path.lower().endswith('.pdf'):
            with open(file_path, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                return ' '.join(page.extract_text() or '' for page in reader.pages)
        elif file_path.lower().endswith('.docx'):
            doc = docx.Document(file_path)
            return '\n'.join(p.text for p in doc.paragraphs)
        elif file_path.lower().endswith('.txt'):
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                return f.read()
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
    return ""

# Check if whole word exists in text (case-insensitive)
def word_in_text(word, text):
    return re.search(rf'\b{re.escape(word)}\b', text, flags=re.IGNORECASE)

# Search logic
def search_files(directory, query, mode):
    results = []
    if not query:
        return results

    if mode == "Search by File Name":
        query_lower = query.lower()
        for root, _, files in os.walk(directory):
            for file in files:
                if query_lower in file.lower():
                    results.append((file, query, os.path.join(root, file), 'N/A'))

    elif mode == "Search by File Content":
        for root, _, files in os.walk(directory):
            for file in files:
                path = os.path.join(root, file)
                content = extract_text_from_file(path)
                if content and word_in_text(query, content):
                    results.append((file, query, path, 'N/A'))

    elif mode == "Semantic Search (AI)":
        query_embedding = model.encode(query, convert_to_tensor=True)
        for root, _, files in os.walk(directory):
            for file in files:
                path = os.path.join(root, file)
                content = extract_text_from_file(path)
                if content:
                    content_embedding = model.encode(content, convert_to_tensor=True)
                    similarity = util.pytorch_cos_sim(query_embedding, content_embedding).item()
                    if similarity > 0.5:  # Threshold for relevance
                        results.append((file, query, path, round(similarity, 3)))
    return results

# GUI
def browse_directory():
    path = filedialog.askdirectory()
    directory_var.set(path)

def on_search():
    tree.delete(*tree.get_children())
    directory = directory_var.get()
    query = search_var.get()
    mode = mode_var.get()
    if not directory or not query:
        messagebox.showwarning("Missing Info", "Please specify both directory and search query.")
        return

    results = search_files(directory, query, mode)
    if not results:
        messagebox.showinfo("No Results", "No matching files found.")
    for file, query, path, score in results:
        tree.insert('', tk.END, values=(file, query, path, score))

# Main Window
root = tk.Tk()
root.title("Intelligent File Search")

# Layout
tk.Label(root, text="Directory:").grid(row=0, column=0, sticky='w', padx=5, pady=5)
directory_var = tk.StringVar()
tk.Entry(root, textvariable=directory_var, width=50).grid(row=0, column=1, padx=5)
tk.Button(root, text="Browse", command=browse_directory).grid(row=0, column=2, padx=5)

tk.Label(root, text="Search:").grid(row=1, column=0, sticky='w', padx=5, pady=5)
search_var = tk.StringVar()
tk.Entry(root, textvariable=search_var, width=50).grid(row=1, column=1, padx=5)

mode_var = tk.StringVar(value="Search by File Name")
mode_dropdown = ttk.Combobox(root, textvariable=mode_var, values=[
    "Search by File Name",
    "Search by File Content",
    "Semantic Search (AI)"
], state="readonly", width=47)
mode_dropdown.grid(row=1, column=2, padx=5)

tk.Button(root, text="Search", command=on_search, bg="lightblue").grid(row=2, column=1, pady=10)

# Results Table
columns = ("File Name", "Search Term", "File Path", "Similarity")
tree = ttk.Treeview(root, columns=columns, show="headings")
for col in columns:
    tree.heading(col, text=col)
    tree.column(col, anchor='w', width=200 if col == "File Path" else 150)
tree.grid(row=3, column=0, columnspan=3, padx=5, pady=5, sticky="nsew")

# Resize behavior
root.grid_rowconfigure(3, weight=1)
root.grid_columnconfigure(1, weight=1)

root.mainloop()

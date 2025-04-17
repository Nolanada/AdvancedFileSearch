import os
import tkinter as tk
from tkinter import filedialog, ttk
from sentence_transformers import SentenceTransformer, util
import numpy as np
import torch
import joblib  # for saving model
import PyPDF2
from docx import Document

# Load or initialize the semantic model
model = SentenceTransformer('all-MiniLM-L6-v2')
joblib.dump(model, 'best_model.h5')  # Save model to 'best_model.h5'

def extract_text_from_file(filepath):
    text = ""
    if filepath.endswith(".txt"):
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            text = f.read()
    elif filepath.endswith(".pdf"):
        try:
            with open(filepath, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                for page in reader.pages:
                    text += page.extract_text()
        except:
            pass
    elif filepath.endswith(".docx"):
        try:
            doc = Document(filepath)
            for para in doc.paragraphs:
                text += para.text
        except:
            pass
    return text

def search_files(directory, query, mode):
    results = []
    query_lower = query.lower()
    if mode == "Search by File Name":
        for root, _, files in os.walk(directory):
            for file in files:
                if query_lower in file.lower():
                    results.append((file, query, os.path.join(root, file), 'N/A'))
    elif mode == "Search by File Content":
        for root, _, files in os.walk(directory):
            for file in files:
                path = os.path.join(root, file)
                content = extract_text_from_file(path).lower()
                if query_lower in content:
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
                    if similarity > 0.4:  # Threshold
                        results.append((file, query, path, round(similarity, 3)))
    return results

# === Tkinter UI ===
def browse_directory():
    folder_selected = filedialog.askdirectory()
    if folder_selected:
        entry_dir.delete(0, tk.END)
        entry_dir.insert(0, folder_selected)

def perform_search():
    directory = entry_dir.get()
    query = entry_query.get()
    mode = combo_mode.get()
    if not directory or not query:
        return
    result_text.delete('1.0', tk.END)
    matches = search_files(directory, query, mode)
    if matches:
        for file, search, path, similarity in matches:
            result_text.insert(tk.END, f"File: {file}\nSearch: {search}\nPath: {path}\nSimilarity: {similarity}\n\n")
    else:
        result_text.insert(tk.END, "No matches found.")

# UI Setup
root = tk.Tk()
root.title("Intelligent File Search")

tk.Label(root, text="Select Directory:").grid(row=0, column=0, sticky="w")
entry_dir = tk.Entry(root, width=60)
entry_dir.grid(row=0, column=1, padx=5)
tk.Button(root, text="Browse", command=browse_directory).grid(row=0, column=2)

tk.Label(root, text="Enter Search Query:").grid(row=1, column=0, sticky="w")
entry_query = tk.Entry(root, width=60)
entry_query.grid(row=1, column=1, padx=5)

combo_mode = ttk.Combobox(root, values=[
    "Search by File Name",
    "Search by File Content",
    "Semantic Search (AI)"
])
combo_mode.current(0)
combo_mode.grid(row=1, column=2)

tk.Button(root, text="Search", command=perform_search).grid(row=2, column=1, pady=10)

result_text = tk.Text(root, width=100, height=30)
result_text.grid(row=3, column=0, columnspan=3, padx=10, pady=10)

root.mainloop()

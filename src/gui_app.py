import tkinter as tk
from tkinter import scrolledtext
from threading import Thread
from analysis.sentiment_analysis import analizar_sentimiento
from analysis.habit_recommender import generar_recomendacion
from utils.memory_manager import clear_memory
import time

# Simula un usuario único (para usar memoria)
USER_ID = 1

class ChatBotGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Asistente de Bienestar 🍎🧠")
        self.root.geometry("500x600")
        self.root.config(bg="#F3F2EF")

        # --- Área de chat ---
        self.chat_area = scrolledtext.ScrolledText(root, wrap=tk.WORD, bg="#ffffff", fg="#333", font=("Segoe UI", 10))
        self.chat_area.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)
        self.chat_area.config(state=tk.DISABLED)

        # --- Entrada de texto ---
        self.entry = tk.Entry(root, bg="#ffffff", font=("Segoe UI", 11))
        self.entry.pack(padx=10, pady=5, fill=tk.X)
        self.entry.bind("<Return>", self.on_send)

        # --- Botones ---
        btn_frame = tk.Frame(root, bg="#F3F2EF")
        btn_frame.pack(pady=5)
        self.send_button = tk.Button(btn_frame, text="Enviar 💬", command=self.on_send, bg="#7FC97F", fg="white", width=12)
        self.send_button.grid(row=0, column=0, padx=5)
        self.reset_button = tk.Button(btn_frame, text="Reset 🧹", command=self.reset_chat, bg="#F15B5B", fg="white", width=12)
        self.reset_button.grid(row=0, column=1, padx=5)

        # --- Mensaje de bienvenida ---
        self.print_bot("🌱 ¡Hola! Soy tu asistente de bienestar alimenticio.\nContame cómo te sentís y te voy a ayudar a mejorar tu alimentación y hábitos. 🍎✨")

    def print_bot(self, text):
        self.chat_area.config(state=tk.NORMAL)
        self.chat_area.insert(tk.END, f"\n🤖 Menta: {text}\n")
        self.chat_area.config(state=tk.DISABLED)
        self.chat_area.see(tk.END)

    def print_user(self, text):
        self.chat_area.config(state=tk.NORMAL)
        self.chat_area.insert(tk.END, f"\n🧍‍♂️ Vos: {text}\n")
        self.chat_area.config(state=tk.DISABLED)
        self.chat_area.see(tk.END)

    def on_send(self, event=None):
        user_text = self.entry.get().strip()
        if not user_text:
            return
        self.print_user(user_text)
        self.entry.delete(0, tk.END)
        Thread(target=self.bot_response, args=(user_text,)).start()

    def bot_response(self, user_text):
        self.print_bot("⏳ Procesando...")

        sentimiento = analizar_sentimiento(user_text)
        time.sleep(0.5)
        respuesta = generar_recomendacion(user_text, sentimiento, USER_ID)

        # Borrar el "procesando" y mostrar respuesta real
        self.chat_area.config(state=tk.NORMAL)
        self.chat_area.delete("end-2l", tk.END)
        self.chat_area.config(state=tk.DISABLED)
        self.print_bot(respuesta)

    def reset_chat(self):
        clear_memory(USER_ID)
        self.chat_area.config(state=tk.NORMAL)
        self.chat_area.delete(1.0, tk.END)
        self.chat_area.config(state=tk.DISABLED)
        self.print_bot("🧹 Conversación reiniciada. Contame cómo te sentís hoy 🍃")

if __name__ == "__main__":
    root = tk.Tk()
    app = ChatBotGUI(root)
    root.mainloop()

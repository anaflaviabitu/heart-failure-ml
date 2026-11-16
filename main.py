import queue
import threading
import tkinter as tk
from tkinter import messagebox, ttk

from modelos import executar


class Aplicacao:
    def __init__(self, janela):
        self.janela = janela
        self.janela.title("Comparação de modelos — Heart Failure Dataset")
        self.janela.geometry("920x500")
        self.janela.configure(bg="#f0f4f8")
        self.eventos = queue.Queue()

        ttk.Label(janela, text="Classificação de desfechos de insuficiência cardíaca", font=("Segoe UI", 15, "bold")).pack(pady=18)
        colunas = ("Modelo", "Acurácia", "Precisão", "Recall", "F1 Score")
        self.tabela = ttk.Treeview(janela, columns=colunas, show="headings", height=10)
        for coluna in colunas:
            self.tabela.heading(coluna, text=coluna)
            self.tabela.column(coluna, width=160, anchor="center")
        self.tabela.pack(fill="both", expand=True, padx=20, pady=12)

        self.botao = ttk.Button(janela, text="Treinar e comparar KNN, SVM e MLP", command=self.iniciar)
        self.botao.pack(pady=10)
        self.status = ttk.Label(janela, text="Pronto para iniciar")
        self.status.pack(pady=10)
        self.janela.after(150, self.processar_eventos)

    def iniciar(self):
        self.botao.configure(state="disabled")
        self.status.configure(text="Treinamento em execução. A otimização pode levar alguns minutos.")
        threading.Thread(target=self.executar_em_segundo_plano, daemon=True).start()

    def executar_em_segundo_plano(self):
        try:
            resultados, detalhes = executar()
            self.eventos.put(("sucesso", resultados, detalhes))
        except Exception as erro:
            self.eventos.put(("erro", str(erro)))

    def processar_eventos(self):
        try:
            while True:
                evento = self.eventos.get_nowait()
                self.botao.configure(state="normal")
                if evento[0] == "erro":
                    self.status.configure(text="Falha no treinamento")
                    messagebox.showerror("Erro", evento[1])
                    continue
                _, resultados, detalhes = evento
                for item in self.tabela.get_children():
                    self.tabela.delete(item)
                for _, linha in resultados.iterrows():
                    valores = [linha["Modelo"]] + [f"{linha[c]:.2%}" for c in ("Acurácia", "Precisão", "Recall", "F1 Score")]
                    self.tabela.insert("", "end", values=valores)
                self.status.configure(text="Concluído. Gráficos e resultados salvos na pasta graficos/.")
                messagebox.showinfo("Concluído", "Treinamento e gráficos gerados com sucesso.")
        except queue.Empty:
            pass
        self.janela.after(150, self.processar_eventos)


def main():
    janela = tk.Tk()
    Aplicacao(janela)
    janela.mainloop()


if __name__ == "__main__":
    main()

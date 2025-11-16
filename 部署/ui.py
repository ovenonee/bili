# -*- coding: utf-8 -*-
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import os
from PIL import Image, ImageTk
import torch
import torch.nn as nn
from torchvision import transforms, models
from sklearn.preprocessing import LabelEncoder
import warnings
import numpy as np
warnings.filterwarnings('ignore')

# ============================================================================
#  模型部署器（必须放在 ModelGUI 之前）
# ============================================================================
class ModelDeployer:
    def __init__(self, model_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 标签编码器
        self.le_views = LabelEncoder()
        self.le_views.fit(['中', '低', '高'])
        self.le_likes = LabelEncoder()
        self.le_likes.fit(['中', '低', '高'])

        # 构建与训练时完全一致的网络
        self.model = self._build_model(len(self.le_views.classes_),
                                       len(self.le_likes.classes_))

        # 加载权重
        state_dict = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state_dict, strict=True)
        self.model.eval()

        # 图像预处理
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    # ---------- 网络结构（与训练时一致） ----------
    def _build_model(self, num_views, num_likes):
        class SimpleMultiTaskModel(nn.Module):
            def __init__(self, num_views, num_likes):
                super().__init__()
                backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
                self.backbone = nn.Sequential(*list(backbone.children())[:-1])  # [B,512]

                # 这一层必须保留，否则权重对不上
                self.label_embedding = nn.Embedding(num_embeddings=32, embedding_dim=32)

                # 输入 512+32=544
                self.views_head = nn.Sequential(
                    nn.Dropout(0.4),
                    nn.Linear(544, 256),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(256, num_views)
                )
                self.likes_head = nn.Sequential(
                    nn.Dropout(0.4),
                    nn.Linear(544, 256),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(256, num_likes)
                )

            # 推理时 label=None 自动补 0 向量
            def forward(self, x, label=None):
                img_feat = self.backbone(x).view(x.size(0), -1)        # [B,512]
                if label is None:
                    label = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
                lab_emb = self.label_embedding(label)                  # [B,32]
                feat = torch.cat([img_feat, lab_emb], dim=1)           # [B,544]
                return self.views_head(feat), self.likes_head(feat)

        return SimpleMultiTaskModel(num_views, num_likes).to(self.device)

    # ---------- 单张图片预测 ----------
    def predict(self, image_path):
        try:
            img = Image.open(image_path).convert('RGB')
            tensor = self.transform(img).unsqueeze(0).to(self.device)
            with torch.no_grad():
                v_out, l_out = self.model(tensor)
                v_pred = torch.argmax(v_out, dim=1).item()
                l_pred = torch.argmax(l_out, dim=1).item()
            return self.le_views.inverse_transform([v_pred])[0], \
                   self.le_likes.inverse_transform([l_pred])[0]
        except Exception as e:
            print("预测错误:", e)
            return None, None

class ModelGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("视频封面热度预测系统")
        self.root.geometry("800x700")
        self.model_deployer = None
        self.setup_ui()          # 这里会调用下方方法

    # ---------- 缺失的就是这个 ----------
    def setup_ui(self):
        # 标题
        title = ttk.Label(self.root, text="视频封面热度预测系统",
                          font=("Arial", 16, "bold"))
        title.pack(pady=10)

        # ---- 模型加载 ----
        load_frame = ttk.LabelFrame(self.root, text="模型加载", padding=10)
        load_frame.pack(fill="x", padx=10, pady=5)

        ttk.Label(load_frame, text="模型路径:").grid(row=0, column=0, sticky="w")
        self.model_path_var = tk.StringVar()
        ttk.Entry(load_frame, textvariable=self.model_path_var,
                  width=50).grid(row=0, column=1, padx=5)
        ttk.Button(load_frame, text="浏览",
                   command=self.browse_model).grid(row=0, column=2, padx=5)
        ttk.Button(load_frame, text="加载模型",
                   command=self.load_model).grid(row=0, column=3, padx=5)

        # ---- 输入信息 ----
        input_frame = ttk.LabelFrame(self.root, text="输入信息", padding=10)
        input_frame.pack(fill="x", padx=10, pady=5)

        ttk.Label(input_frame, text="封面图片:").grid(row=0, column=0, sticky="w")
        self.image_path_var = tk.StringVar()
        ttk.Entry(input_frame, textvariable=self.image_path_var,
                  width=50).grid(row=0, column=1, padx=5)
        ttk.Button(input_frame, text="浏览",
                   command=self.browse_image).grid(row=0, column=2, padx=5)

        ttk.Label(input_frame, text="实际播放量:").grid(row=1, column=0, sticky="w", pady=5)
        self.views_var = tk.StringVar()
        ttk.Entry(input_frame, textvariable=self.views_var,
                  width=20).grid(row=1, column=1, sticky="w", pady=5)

        ttk.Label(input_frame, text="实际点赞量:").grid(row=2, column=0, sticky="w", pady=5)
        self.likes_var = tk.StringVar()
        ttk.Entry(input_frame, textvariable=self.likes_var,
                  width=20).grid(row=2, column=1, sticky="w", pady=5)

        ttk.Button(input_frame, text="预测",
                   command=self.predict_and_evaluate).grid(row=3, column=0,
                                                           columnspan=3, pady=10)

        # ---- 结果显示 ----
        result_frame = ttk.LabelFrame(self.root, text="预测结果", padding=10)
        result_frame.pack(fill="both", expand=True, padx=10, pady=5)

        self.image_label = ttk.Label(result_frame, text="图片将在此显示",
                                     anchor="center")
        self.image_label.pack(pady=10)

        self.result_text = tk.Text(result_frame, height=8, width=70)
        self.result_text.pack(pady=10, padx=10, fill="both", expand=True)

        scroll = ttk.Scrollbar(result_frame, orient="vertical",
                               command=self.result_text.yview)
        scroll.pack(side="right", fill="y", padx=(0, 10))
        self.result_text.configure(yscrollcommand=scroll.set)

        ttk.Button(result_frame, text="清空结果",
                   command=self.clear_results).pack(pady=5)

    # ---------- 以下方法同之前 ----------
    def browse_model(self):
        path = filedialog.askopenfilename(
            title="选择模型文件",
            filetypes=[("PyTorch模型", "*.pth"), ("所有文件", "*.*")])
        if path:
            self.model_path_var.set(path)

    def browse_image(self):
        path = filedialog.askopenfilename(
            title="选择封面图片",
            filetypes=[("图片文件", "*.jpg *.jpeg *.png *.bmp"),
                       ("所有文件", "*.*")])
        if path:
            self.image_path_var.set(path)
            self.display_image(path)

    def display_image(self, path):
        try:
            img = Image.open(path).resize((200, 200), Image.LANCZOS)
            self.photo = ImageTk.PhotoImage(img)
            self.image_label.config(image=self.photo, text="")
        except Exception as e:
            self.image_label.config(image="", text=f"无法显示图片: {e}")

    def load_model(self):
        path = self.model_path_var.get()
        if not os.path.isfile(path):
            messagebox.showerror("错误", "模型文件不存在！")
            return
        try:
            self.model_deployer = ModelDeployer(path)
            messagebox.showinfo("成功", "模型加载成功！")
        except Exception as e:
            messagebox.showerror("错误", f"模型加载失败: {e}")

    def predict_and_evaluate(self):
        if not self.model_deployer:
            messagebox.showerror("错误", "请先加载模型！")
            return
        img_path = self.image_path_var.get()
        if not os.path.isfile(img_path):
            messagebox.showerror("错误", "图片文件不存在！")
            return
        try:
            actual_v = int(self.views_var.get())
            actual_l = int(self.likes_var.get())
        except ValueError:
            messagebox.showerror("错误", "请输入有效数字！")
            return

        pred_v, pred_l = self.model_deployer.predict(img_path)
        if pred_v is None:
            messagebox.showerror("错误", "预测失败！")
            return

        # 实际类别计算同之前
        import numpy as np
        log_v = np.log10(actual_v + 1)
        log_l = np.log10(actual_l + 1)
        true_v = '低' if log_v < 5 else '中' if log_v < 7 else '高'
        true_l = '低' if log_l < 4 else '中' if log_l < 6 else '高'
        match_v = pred_v == true_v
        match_l = pred_l == true_l

        report = f"""
预测结果分析
{'='*50}
图片路径: {img_path}

实际数据:
- 播放量: {actual_v:,} (类别: {true_v})
- 点赞量: {actual_l:,} (类别: {true_l})

预测结果:
- 播放量: {pred_v} {'✅' if match_v else '❌'}
- 点赞量: {pred_l} {'✅' if match_l else '❌'}

整体预测: {'✅ 正确' if match_v and match_l else '❌ 错误'}
""".strip()

        self.result_text.delete(1.0, tk.END)
        self.result_text.insert("1.0", report)
        self.display_image(img_path)

    def clear_results(self):
        self.result_text.delete(1.0, tk.END)
        self.image_label.config(image="", text="图片将在此显示")


# ---------- main ----------
def main():
    root = tk.Tk()
    ModelGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
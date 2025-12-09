# TorchQuad Hesap Makinesi

Bu proje, **PyTorch**, **SymPy**, **TorchQuad** ve **Tkinter** kullanarak hem sembolik hem de sayısal integraller hesaplayabilen bir masaüstü uygulamasıdır. Kullanıcıdan matematiksel bir fonksiyon ve alt/üst sınırlar alarak farklı integral yöntemleriyle sonucu hesaplar.

---

## 🚀 Özellikler

- 🧮 **Sembolik integral hesaplama** (SymPy)
- 🔢 **Sayısal integral yöntemleri**:
  - Monte Carlo (TorchQuad)
  - Simpson (TorchQuad)
  - Trapezoid (TorchQuad)
  - Boole (TorchQuad)
  - SciPy (karşılaştırma için)
- 🖥️ **Basit ve kullanıcı dostu Tkinter arayüzü**
- ⚠️ Hata yakalama, giriş doğrulama ve log sistemi

---

## 📦 Kullanılan Kütüphaneler

Aşağıdaki Python paketlerine ihtiyaç vardır:

- `torch`
- `sympy`
- `numpy`
- `tkinter` (Python ile birlikte gelir)
- `scipy`
- `torchquad`
- `logging`

Kurulum:

```bash
pip install torch sympy numpy scipy torchquad

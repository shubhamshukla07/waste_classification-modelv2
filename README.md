# 🧠 Waste Classification using CNN & Transfer Learning

A deep learning-powered web app that classifies **waste images** into one of **9 categories** using a Convolutional Neural Network (CNN) with **Transfer Learning** (TensorFlow & Keras). The model achieves **89% accuracy** and is deployed using [Streamlit](https://streamlit.io).

## ♻️ Supported Waste Categories

- Trash
- Paper Waste
- Textile Waste
- Organic Waste
- Metal Waste
- Plastic Waste
- Cardboard
- Glass Waste
- E-Waste

## 🚀 Live Demo

👉 [Click here to try the app](https://wasteclassification-modelv2-azn8fegsadqvixbckhhwen.streamlit.app/)

## 🧠 Model Overview

- Built with **TensorFlow** and **Keras**
- Based on a **pretrained model (Transfer Learning)** for faster convergence and better accuracy
- Custom dataset with 9 labeled waste classes
- Achieved **89% validation accuracy**

## 📦 Tech Stack

- Python
- TensorFlow / Keras
- Streamlit (for the web app)
- NumPy, Pillow, OpenCV

## 📁 Project Structure

├── hi.py # Main Streamlit app script
├── model/ # Saved TensorFlow/Keras trained model files
├── dataset/ # (Optional) Dataset images used for training
├── requirements.txt # List of Python dependencies
└── README.md # Project overview and instructions



## 🛠️ How to Run Locally

1. Clone the repo:

```bash
git clone https://github.com/shubhamshukla07/wasteclassification-modelv2.git
cd wasteclassification-modelv2
pip install -r requirements.txt
streamlit run hi.py


## 📜 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## 🤝 Contributing

Contributions are welcome! If you'd like to contribute, please:

1. Fork the repository
2. Create a new branch (`git checkout -b feature-name`)
3. Commit your changes (`git commit -m 'Add some feature'`)
4. Push to the branch (`git push origin feature-name`)
5. Open a Pull Request

Please make sure your code follows the existing style and includes relevant tests if applicable.

---

## 📬 Contact

For any questions or feedback, feel free to reach out:

- **Shubham Shukla**  
- Email: shubhamshukla223311@gmail.com 
- GitHub: [https://github.com/shubhamshukla07](https://github.com/shubhamshukla07)

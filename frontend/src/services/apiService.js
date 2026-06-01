// frontend/src/services/apiService.js

const BASE_URL = "http://127.0.0.1:8000/api";

// Genel API Çağrı Fonksiyonu
export const fetchAllDashboardDataFromBackend = async (modelName, symbol) => {
    try {
        const response = await fetch(`${BASE_URL}/dashboard?model_name=${encodeURIComponent(modelName)}&symbol=${encodeURIComponent(symbol)}`);
        if (!response.ok) throw new Error(`Backend API Hatası: ${response.status}`);
        return await response.json();
    } catch (error) {
        console.error("Backend sunucusuna bağlanılamadı:", error);
        throw error;
    }
};

// Sabit Dropdown listeleri (Şimdilik arayüz kırılmasın diye mock veriyoruz)
export const fetchSymbols = async () => ["THYAO", "ACSEL", "TTKOM", "ASELS", "AKBNK"];
// frontend/src/services/apiService.js içindeki ilgili fonksiyonu şununla değiştirin:
export const fetchModels = async () => [
    { id: "MLP DQN", name: "MLP DQN" },
    { id: "LSTM DQN", name: "LSTM DQN" },
    { id: "GRU DQN", name: "GRU DQN" },
    { id: "CNN DQN", name: "CNN DQN" },
    { id: "Dueling DQN", name: "Dueling DQN" }
];
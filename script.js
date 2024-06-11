const products = [
    {
        "nama": "Malkist Roma Crackers",
        "deskripsi": "Crackers gurih dengan taburan gula yang renyah.",
        "harga": 15000,
        "berat": 150,
        "kandungan_kkal": 480
    },
    {
        "nama": "Malkist Roma Belgian Chocolate",
        "deskripsi": "Crackers dengan lapisan cokelat Belgia yang lezat.",
        "harga": 18000,
        "berat": 125,
        "kandungan_kkal": 540
    },
    {
        "nama": "Malkist Roma Keju Manis",
        "deskripsi": "Crackers dengan rasa keju manis yang menggugah selera.",
        "harga": 16000,
        "berat": 130,
        "kandungan_kkal": 520
    },
    {
        "nama": "Malkist Roma Abon",
        "deskripsi": "Crackers dengan taburan abon sapi yang gurih.",
        "harga": 17000,
        "berat": 140,
        "kandungan_kkal": 510
    },
    {
        "nama": "Malkist Roma Cokelat Kelapa",
        "deskripsi": "Crackers dengan rasa cokelat kelapa yang unik.",
        "harga": 18000,
        "berat": 135,
        "kandungan_kkal": 530
    },
    {
        "nama": "Malkist Roma Sandwich Chocolate",
        "deskripsi": "Sandwich crackers dengan isian cokelat manis.",
        "harga": 19000,
        "berat": 140,
        "kandungan_kkal": 550
    },
    {
        "nama": "Malkist Roma Sandwich Peanut Butter",
        "deskripsi": "Sandwich crackers dengan isian selai kacang yang kaya.",
        "harga": 20000,
        "berat": 145,
        "kandungan_kkal": 560
    },
    {
        "nama": "Malkist Roma Biskuit Kelapa",
        "deskripsi": "Biskuit dengan rasa kelapa yang menggoda.",
        "harga": 15000,
        "berat": 150,
        "kandungan_kkal": 480
    },
    {
        "nama": "Malkist Roma Biskuit Kelapa Kopyor",
        "deskripsi": "Biskuit dengan cita rasa kelapa kopyor.",
        "harga": 16000,
        "berat": 155,
        "kandungan_kkal": 490
    },
    {
        "nama": "Malkist Sari Gandum",
        "deskripsi": "Biskuit gandum dengan rasa alami dan sehat.",
        "harga": 14000,
        "berat": 160,
        "kandungan_kkal": 460
    },
    {
        "nama": "Oreo Original",
        "deskripsi": "Biskuit hitam dengan krim putih klasik.",
        "harga": 12000,
        "berat": 100,
        "kandungan_kkal": 470
    },
    {
        "nama": "Oreo Chocolate",
        "deskripsi": "Oreo dengan rasa cokelat yang lebih intens.",
        "harga": 13000,
        "berat": 100,
        "kandungan_kkal": 480
    },
    {
        "nama": "Oreo Ice Cream",
        "deskripsi": "Oreo dengan rasa es krim yang menyegarkan.",
        "harga": 14000,
        "berat": 100,
        "kandungan_kkal": 490
    },
    {
        "nama": "Oreo Red Velvet",
        "deskripsi": "Oreo dengan rasa red velvet dan krim keju.",
        "harga": 15000,
        "berat": 100,
        "kandungan_kkal": 500
    },
    {
        "nama": "Oreo Fizzy",
        "deskripsi": "Oreo dengan rasa soda yang unik.",
        "harga": 15000,
        "berat": 100,
        "kandungan_kkal": 500
    },
    {
        "nama": "Tango Vanilla Delight",
        "deskripsi": "Wafer dengan rasa vanilla yang nikmat.",
        "harga": 12000,
        "berat": 125,
        "kandungan_kkal": 450
    },
    {
        "nama": "Tango Royal Chocolate",
        "deskripsi": "Wafer dengan lapisan cokelat royal.",
        "harga": 13000,
        "berat": 125,
        "kandungan_kkal": 460
    },
    {
        "nama": "Tango Sassy Strawberry",
        "deskripsi": "Wafer dengan rasa stroberi yang segar.",
        "harga": 12000,
        "berat": 125,
        "kandungan_kkal": 450
    },
    {
        "nama": "Silverqueen",
        "deskripsi": "Cokelat kacang klasik dengan rasa yang istimewa.",
        "harga": 25000,
        "berat": 100,
        "kandungan_kkal": 520
    },
    {
        "nama": "Toblerone",
        "deskripsi": "Cokelat Swiss dengan nougat dan almond yang ikonik.",
        "harga": 30000,
        "berat": 100,
        "kandungan_kkal": 530
    },
    {
        "nama": "Delfi",
        "deskripsi": "Cokelat dengan rasa lezat dan creamy.",
        "harga": 20000,
        "berat": 100,
        "kandungan_kkal": 510
    },
    {
        "nama": "Cadburry",
        "deskripsi": "Cokelat susu yang creamy dan lezat.",
        "harga": 27000,
        "berat": 100,
        "kandungan_kkal": 500
    }
];

const imageInput = document.getElementById('imageInput');
const scanButton = document.getElementById('scanButton');
const loading = document.getElementById('loading');
const result = document.getElementById('result');

scanButton.addEventListener('click', () => {
    const file = imageInput.files[0];
    if (!file) {
        alert('Please upload an image first.');
        return;
    }

    loading.classList.remove('d-none');
    result.classList.add('d-none');

    // Simulating image recognition process with a timeout
    setTimeout(() => {
        loading.classList.add('d-none');
        displayResult(products[0]); // For demo, we always return the first product
    }, 2000);
});

function displayResult(product) {
    document.getElementById('productName').textContent = product.nama;
    document.getElementById('productDescription').textContent = product.deskripsi;
    document.getElementById('productPrice').textContent = product.harga;
    document.getElementById('productWeight').textContent = product.berat;
    document.getElementById('productKkal').textContent = product.kandungan_kkal;
    result.classList.remove('d-none');
}

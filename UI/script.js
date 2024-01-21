document.addEventListener("DOMContentLoaded", function () {
    changeImage('lstm', 'pres', 'test');
    updatePredictionTitle('lstm');
});

function changeModel(model) {
    document.getElementById("image-selector").selectedIndex = 0;
    changeImage(model, document.getElementById("image-selector").value, document.getElementById("data-selector").value);
    updatePredictionTitle(model);
}

function updatePredictionTitle(model) {
    const predictionTitleElement = document.getElementById("prediction-title");
    if (model === 'lstm') {
        predictionTitleElement.textContent = "Future prediction for 1h (24h history)";
    } else if (model === 'prophet') {
        predictionTitleElement.textContent = "Future prediction for 48h";
    }else if (model === 'seq2seq') {
        predictionTitleElement.textContent = "Future prediction for 1h (24h history)";
    }
}

function changeData(data) {
    document.getElementById("image-selector").selectedIndex = 0;
    changeImage(document.getElementById("model-selector").value, document.getElementById("image-selector").value, data);
}

function changeImage(model, imgName, data) {
    const imageUrl = `img/${model}/${data}/${imgName}.png`;

    const tempImage = new Image();
    tempImage.src = imageUrl;

    tempImage.onload = function () {
        document.getElementById("displayed-image").src = imageUrl;
        
        if(model === 'prophet')
            jsonFilePath = `risk_data/prophet.json`
        else if(model === 'lstm')
            jsonFilePath = `risk_data/lstm.json`
        else
            jsonFilePath =`risk_data/seqtoseq.json`

        fetch(jsonFilePath)
            .then(response => response.json())
            .then(data => {
                updateDiseaseRisk(data);
            })
            .catch(error => console.error("Error fetching JSON:", error));
    };
}


function updateDiseaseRisk(data) {
    const earlyBlightRisk = parseFloat(data.EarlyBlight);
    const grayMoldRisk = parseFloat(data.GrayMold);
    const lateBlightRisk = parseFloat(data.LateBlight);
    const lateMoldRisk = parseFloat(data.LateMold);
    const powderyMildewRisk = parseFloat(data.PowderyMildew);

    const formattedEarlyBlightRisk = earlyBlightRisk.toFixed(2);
    const formattedGrayMoldRisk = grayMoldRisk.toFixed(2);
    const formattedLateBlightRisk = lateBlightRisk.toFixed(2);
    const formattedLateMoldRisk = lateMoldRisk.toFixed(2);
    const formattedPowderyMildewRisk = powderyMildewRisk.toFixed(2);

    document.getElementById("early-blight-risk").textContent = `Early Blight: ${formattedEarlyBlightRisk}%`;
    document.getElementById("gray-mold-risk").textContent = `Gray Mold: ${formattedGrayMoldRisk}%`;
    document.getElementById("late-blight-risk").textContent = `Late Blight: ${formattedLateBlightRisk}%`;
    document.getElementById("late-mold-risk").textContent = `Late Mold: ${formattedLateMoldRisk}%`;
    document.getElementById("powdery-mildew-risk").textContent = `Powdery Mildew: ${formattedPowderyMildewRisk}%`;
}


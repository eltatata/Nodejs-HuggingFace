import "dotenv/config";
import { HfInference } from "@huggingface/inference";

const hf = new HfInference(process.env.HUGGINGFACE_TOKEN)

// -- image to text --
const imageUrl = "https://miro.medium.com/v2/resize:fit:1400/1*PjmGdECv7LAlQW6hm81K6w.png";

try {
    const response = await fetch(imageUrl);
    const blob = await response.blob();

    const text = await hf.imageToText({
        data: blob,
        model: 'Salesforce/blip-image-captioning-large'
    });

    console.log(text);
} catch (error) {
    console.log(error);
}

// -- translation --
try {
    const translation = await hf.translation({
        model: 'facebook/mbart-large-50-many-to-many-mmt',
        inputs: "This is a test text to be translated by the model",
        parameters: {
            "src_lang": "en_XX",
            "tgt_lang": "de_DE"
        }
    })

    console.log(translation)
} catch (error) {
    console.log(error);
}
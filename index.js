import "dotenv/config";
import { HfInference } from "@huggingface/inference";

const hf = new HfInference(process.env.HUGGINGFACE_TOKEN)

// -- image to text --
const model = 'Salesforce/blip-image-captioning-large';
const imageUrl = "https://i.blogs.es/662997/agujero-blanco/1366_2000.jpg";

try {
    const response = await fetch(imageUrl);
    const blob = await response.blob();

    const text = await hf.imageToText({
        data: blob,
        model
    });

    console.log(text);
} catch (error) {
    console.log(error);
}
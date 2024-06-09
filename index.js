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

// -- text generation (chat completion) --
try {
    let messages = [{ role: "user", content: "What is the largest land animal" }]
    const out = await hf.chatCompletion({
        model: "mistralai/Mistral-7B-Instruct-v0.2",
        messages,
        max_tokens: 500,
        temperature: 0.1,
        seed: 0,
    });
    messages = [...messages, out.choices[0].message]


    messages = [...messages, { role: "user", content: "What is the jaguar and what is its top speed and how does it differ from a cheeta?" }]
    const out2 = await hf.chatCompletion({
        model: "mistralai/Mistral-7B-Instruct-v0.2",
        messages,
        max_tokens: 500,
        temperature: 0.1,
        seed: 0,
    });
    messages = [...messages, out2.choices[0].message]

    console.log("MESSAGE HISTORY")
    console.log(messages)
} catch (error) {
    console.log(error);
}

// -- streaming --
try {
    const messages = [{
        role: "user",
        content: "What are Newton's laws and their formulas?"
    }];

    let out = "";

    for await (const chunk of hf.chatCompletionStream({
        model: "mistralai/Mistral-7B-Instruct-v0.2",
        messages,
        max_tokens: 500,
        temperature: 0.1,
        seed: 0,
    })) {
        if (chunk.choices && chunk.choices.length > 0) {
            out += chunk.choices[0].delta.content;
            console.log(out)
        }
    }
} catch (error) {
    console.log(error);
}
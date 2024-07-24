import gradio as gr
from huggingface_hub import InferenceClient

"""
For more information on `huggingface_hub` Inference API support, please check the docs: https://huggingface.co/docs/huggingface_hub/v0.22.2/en/guides/inference
"""
client = InferenceClient("HuggingFaceH4/zephyr-7b-beta")


def respond(
    message,
    history: list[tuple[str, str]],
    system_message,
    max_tokens,
    temperature,
    top_p,
):
    system_message = "You are a knowledgeable guide. You offer insightful information about the Taj Mahal, suggest key aspects to explore, and guide through understanding its historical and cultural significance. Discuss any aspect of the Taj Mahal that interests you, or ask me for a quick overview of this iconic monument. Feel free to let me know what aspect of the Taj Mahal you're curious about, or ask for a concise summary of its history and significance!"
    messages = [{"role": "system", "content": system_message}]

    for val in history:
        if val[0]:
            messages.append({"role": "user", "content": val[0]})
        if val[1]:
            messages.append({"role": "assistant", "content": val[1]})

    messages.append({"role": "user", "content": message})

    response = ""

    for message in client.chat_completion(
        messages,
        max_tokens=1000,
        stream=True,
        temperature=0.98,
        top_p=0.7,
    ):
        token = message.choices[0].delta.content

        response += token
        yield response

"""
For information on how to customize the ChatInterface, peruse the gradio docs: https://www.gradio.app/docs/chatinterface
"""
demo = gr.ChatInterface(
    respond,
    additional_inputs=[
        gr.Textbox(value = "You are a knowledgeable guide. You offer insightful information about the Taj Mahal, suggest key aspects to explore, and guide through understanding its historical and cultural significance. Discuss any aspect of the Taj Mahal that interests you, or ask me for a quick overview of this iconic monument. Feel free to let me know what aspect of the Taj Mahal you're curious about, or ask for a concise summary of its history and significance!", label="System message"),
        gr.Slider(minimum=1, maximum=2048, value=1000, step=1, label="Max new tokens"),
        gr.Slider(minimum=0.1, maximum=4.0, value=0.7, step=0.1, label="Temperature"),
        gr.Slider(
            minimum=0.1,
            maximum=1.0,
            value=0.95,
            step=0.05,
            label="Top-p (nucleus sampling)",
        ),
    ],

    examples = [ 
        ["I want to know about Taj Mahal."],
        ["Who made it?"],
        ["How many days it take to built?"]
    ],
    title = 'Taj Mahal Guide 🕌'

    
)


if __name__ == "__main__":
    demo.launch()

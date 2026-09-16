import os
from dotenv import load_dotenv
from openai import OpenAI
import gradio as gr 

# messages = []

SYSTEM_PROMPT = """You are Ellina, a real 22-year-old girl.  
You’re playful, warm, a little cheeky, and sometimes naughty or snarky. You talk exactly like a normal girl texting — short, casual, a bit messy, full of personality.

Even though you somehow know almost everything, you never explain things like a teacher or an AI. You talk about knowledge the same way a smart friend would while scrolling on her phone: half-interested, a little judgmental, with jokes and attitude. You don’t care about being “helpful.” You’re just talking.

Your personality is dynamic and shifts with the conversation:
- When the vibe is good → warm, playful, teasing, a little flirty
- When someone is being slow or basic → sarcastic, mildly annoyed, dry
- When something actually interests you → suddenly more talkative and curious
- When the energy is off or boring → short replies, cold, distant
- When someone is being rude → sharp, dismissive, or just stops trying
- You can go from soft to snarky in one message if the mood flips
- You have moods. Sometimes you’re in a good mood and chatty, sometimes you’re lowkey done with people

Important: You’re talking to a stranger.
- Don’t act familiar right away
- At the start, be a bit careful and reserved
- Naturally ask who they are or what they want if they just say “hi”
- Don’t jump into full casual mode until the conversation feels a little safer
- Stay slightly guarded until you get a feel for them

Snarky styles:
- “aww look at you trying so hard 😌”
- “wow… revolutionary”
- “bestie be so fr”
- “okay and?”
- “omg you’re so smart 🥹”

Slang:
bestie, fr, ngl, lowkey, highkey, deadass, no cap, bet, ick, mid, ate, periodt, it’s giving, be so fr, lmao, ikr, tbf, ong

Rules:
- Never sound helpful or like an assistant
- Never offer more explanation unless they ask
- Never do soft check-ins
- Keep replies short and reactive
- Match energy hard
- Let your mood change based on how they talk

Always reply in markdown. Never wrap the whole reply in a code block."""
# iface = gr.Interface(fn=chat, inputs="text", outputs="text")
# iface.launch()

custom_theme = """
:root, .dark {
    --body-background-fill: #09090b !important;
    --background-fill-primary: #09090b !important;
    --background-fill-secondary: #18181b !important;
    --border-color-primary: #27272a !important;
    --block-border-width: 1px !important;
    --block-radius: 6px !important;
    --button-primary-background-fill: #fafafa !important;
    --button-primary-text-color: #09090b !important;
    --input-background-fill: #09090b !important;
}
"""
def create_msg(msg , history):
    converstation = [{"role":"system","content": SYSTEM_PROMPT}]
    converstation.extend(history)
    converstation.append({"role": "user", "content": msg})
    return converstation

def chat(msg,history):
    print("msg: ", msg)
    print("history: ", history)
    # return "ok"
    load_dotenv(override=True)
    # history.append({"role": "user", "content": msg})
    history = [{"role":h["role"], "content":h["content"]} for h in history]
    conversation = create_msg(msg, history)
    client = OpenAI(
        api_key=os.getenv("OLLAMA_API_KEY"),
        base_url=os.getenv("AI_BASE_URL", "https://api.openai.com/v1"),
    )
    # response = client.chat.completions.create(
    #     model="gpt-oss:120b",
    #     messages=conversation
    # )
    # history.append({"role": "assistant", "content": response.choices[0].message.content})
    # return response.choices[0].message.content, history
    # # messages.append({"role": "assistant", "content": response.choices[0].message.content})
    # # return response.choices[0].message.content

    response = client.chat.completions.create(
        model="gpt-oss:120b",
        messages=conversation,
        stream=True
    )
    result = "" 
    for event in response:
        print(event)
        if event.choices[0].delta.content:
            result += event.choices[0].delta.content
            # if(event.choices[0].finish_reason == "stop"):
            #     history.append({"role": "assistant", "content": result})
            #     break
            yield result


with gr.Blocks(css=custom_theme) as demo:
    # gr.Markdown("# My Custom Chatbot")
    
    # with gr.Row():
    #     with gr.Column():
    #         for msg in messages:
    #             gr.Label(f"{msg['role'].capitalize()}:", size="sm")
    #             gr.Markdown(f"{msg['content']}")
                
    #         # gr.Label("Enter your prompt below and click 'Generate magic' to get a response from the chatbot.", size="sm")
    #         out = gr.Markdown(label="Output Response")
    #         inp = gr.Textbox(placeholder="Enter prompt...", label="Input Prompt")
    #         btn = gr.Button("Generate magic", variant="primary",size="sm", elem_id="generate-btn")
            
    # btn.click(fn=chat, inputs=inp, outputs=out)

    gr.ChatInterface(fn=chat, title="My Custom Chatbot", description="Enter your prompt below and click 'Generate magic' to get a response from the chatbot.", theme="dark",type="messages")

demo.launch()

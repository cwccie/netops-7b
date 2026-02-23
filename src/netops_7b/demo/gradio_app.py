"""Gradio web interface for NetOps-7B.

Provides an interactive demo with three modes:
  1. Configuration Generation — generate vendor-specific configs
  2. Troubleshooting — diagnose network issues
  3. Command Explanation — explain CLI commands
"""

from __future__ import annotations

from typing import Any

import gradio as gr


SYSTEM_PROMPT = (
    "You are NetOps-7B, an expert network operations assistant specialized in "
    "Cisco, Juniper, and Arista network equipment. You help network engineers with "
    "configuration generation, troubleshooting, and command explanation. "
    "Provide accurate, detailed, and vendor-specific responses."
)

CONFIG_EXAMPLES = [
    "Generate a Cisco IOS configuration for OSPF area 0 on interface GigabitEthernet0/0 with network 10.0.0.0/24.",
    "Write a Junos BGP configuration for neighbor 10.1.1.1 in AS 65001 with MD5 authentication.",
    "Create an Arista EOS VXLAN VTEP configuration for VLAN 100 mapped to VNI 10100.",
    "Configure a Cisco NX-OS vPC domain 10 with peer-keepalive on management interface.",
]

TROUBLESHOOTING_EXAMPLES = [
    "BGP session stuck in Active state between two directly-connected routers. How do I troubleshoot?",
    "OSPF neighbor is stuck in EXSTART/EXCHANGE state. What are the likely causes?",
    "Users report intermittent packet loss across the WAN. All interfaces show up/up. How do I isolate the issue?",
    "VXLAN fabric: remote MAC addresses not being learned on leaf switches. Local learning works fine.",
]

EXPLANATION_EXAMPLES = [
    "Explain the Cisco command: show ip bgp summary",
    "Explain the Junos command: show route table inet.0 protocol bgp",
    "Explain the output fields of 'show interfaces' on Cisco IOS",
    "What does 'show ip ospf database router' tell you and how do you read it?",
]


class NetOpsDemo:
    """Gradio demo application for NetOps-7B."""

    def __init__(
        self,
        model_path: str = "cwccie/netops-7b",
        max_new_tokens: int = 2048,
        temperature: float = 0.3,
    ):
        self.model_path = model_path
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.model = None
        self.tokenizer = None

    def load_model(self) -> None:
        """Load the model for inference."""
        from unsloth import FastLanguageModel

        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.model_path,
            max_seq_length=4096,
            load_in_4bit=True,
        )
        FastLanguageModel.for_inference(self.model)

    def generate(
        self,
        instruction: str,
        mode: str = "general",
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """Generate a response."""
        if self.model is None:
            return "Error: Model not loaded. Run load_model() first."

        temp = temperature or self.temperature
        max_tok = max_tokens or self.max_new_tokens

        # Adjust system prompt based on mode
        mode_prompts = {
            "config": SYSTEM_PROMPT + " Focus on generating correct, production-ready configurations.",
            "troubleshoot": SYSTEM_PROMPT + " Follow a systematic troubleshooting approach from Layer 1 upward.",
            "explain": SYSTEM_PROMPT + " Provide detailed explanations with field descriptions and usage context.",
            "general": SYSTEM_PROMPT,
        }
        system = mode_prompts.get(mode, SYSTEM_PROMPT)

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": instruction},
        ]

        input_text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(input_text, return_tensors="pt").to(self.model.device)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_tok,
            temperature=temp,
            do_sample=temp > 0,
            top_p=0.9,
            repetition_penalty=1.1,
        )

        generated = outputs[0][inputs["input_ids"].shape[-1]:]
        return self.tokenizer.decode(generated, skip_special_tokens=True)

    def build_interface(self) -> gr.Blocks:
        """Build the Gradio interface."""

        with gr.Blocks(
            title="NetOps-7B Demo",
            theme=gr.themes.Soft(),
            css="""
            .header { text-align: center; margin-bottom: 20px; }
            .footer { text-align: center; margin-top: 20px; font-size: 0.85em; color: #666; }
            """,
        ) as demo:
            gr.Markdown(
                """
                # NetOps-7B: Network Operations Assistant
                *A QLoRA fine-tuned Qwen 2.5 7B model for network engineering tasks*

                **Model**: [cwccie/netops-7b](https://huggingface.co/cwccie/netops-7b) |
                **Author**: Corey A. Wade (CCIE #14124)
                """,
                elem_classes="header",
            )

            with gr.Tabs():
                # Tab 1: Configuration Generation
                with gr.Tab("Configuration Generation"):
                    gr.Markdown("### Generate vendor-specific network configurations")
                    config_input = gr.Textbox(
                        label="Describe the configuration you need",
                        placeholder="e.g., Generate a Cisco IOS BGP configuration for AS 65001...",
                        lines=3,
                    )
                    config_vendor = gr.Dropdown(
                        choices=["Auto-detect", "Cisco IOS", "Cisco NX-OS", "Cisco IOS-XR",
                                 "Juniper Junos", "Arista EOS"],
                        value="Auto-detect",
                        label="Target Platform",
                    )
                    config_btn = gr.Button("Generate Config", variant="primary")
                    config_output = gr.Code(label="Generated Configuration", language=None, lines=20)
                    gr.Examples(examples=CONFIG_EXAMPLES, inputs=config_input)

                    config_btn.click(
                        fn=lambda text, vendor: self.generate(
                            f"[{vendor}] {text}" if vendor != "Auto-detect" else text,
                            mode="config",
                        ),
                        inputs=[config_input, config_vendor],
                        outputs=config_output,
                    )

                # Tab 2: Troubleshooting
                with gr.Tab("Troubleshooting"):
                    gr.Markdown("### Diagnose and resolve network issues")
                    trouble_input = gr.Textbox(
                        label="Describe the network issue",
                        placeholder="e.g., OSPF neighbor stuck in EXSTART state...",
                        lines=4,
                    )
                    trouble_btn = gr.Button("Diagnose", variant="primary")
                    trouble_output = gr.Markdown(label="Diagnosis & Resolution")
                    gr.Examples(examples=TROUBLESHOOTING_EXAMPLES, inputs=trouble_input)

                    trouble_btn.click(
                        fn=lambda text: self.generate(text, mode="troubleshoot"),
                        inputs=trouble_input,
                        outputs=trouble_output,
                    )

                # Tab 3: Command Explanation
                with gr.Tab("Command Explanation"):
                    gr.Markdown("### Understand CLI commands and their output")
                    explain_input = gr.Textbox(
                        label="Enter a command to explain",
                        placeholder="e.g., show ip bgp summary",
                        lines=2,
                    )
                    explain_btn = gr.Button("Explain", variant="primary")
                    explain_output = gr.Markdown(label="Explanation")
                    gr.Examples(examples=EXPLANATION_EXAMPLES, inputs=explain_input)

                    explain_btn.click(
                        fn=lambda text: self.generate(text, mode="explain"),
                        inputs=explain_input,
                        outputs=explain_output,
                    )

                # Tab 4: Settings
                with gr.Tab("Settings"):
                    gr.Markdown("### Generation Parameters")
                    temp_slider = gr.Slider(
                        minimum=0.0, maximum=1.0, value=0.3, step=0.05,
                        label="Temperature",
                        info="Lower = more deterministic, Higher = more creative",
                    )
                    max_tokens_slider = gr.Slider(
                        minimum=256, maximum=4096, value=2048, step=256,
                        label="Max Tokens",
                    )

                    def update_settings(temp, max_tok):
                        self.temperature = temp
                        self.max_new_tokens = max_tok
                        return f"Settings updated: temperature={temp}, max_tokens={max_tok}"

                    settings_btn = gr.Button("Apply Settings")
                    settings_output = gr.Textbox(label="Status")
                    settings_btn.click(
                        fn=update_settings,
                        inputs=[temp_slider, max_tokens_slider],
                        outputs=settings_output,
                    )

            gr.Markdown(
                """
                ---
                *Built by Corey A. Wade (CCIE #14124, CISSP) | NetOps-7B is a fine-tuned Qwen 2.5 7B model
                for network operations. Not intended for production use without human review.*
                """,
                elem_classes="footer",
            )

        return demo

    def launch(self, **kwargs: Any) -> None:
        """Build and launch the demo."""
        demo = self.build_interface()
        demo.launch(**kwargs)


def create_app(model_path: str = "cwccie/netops-7b") -> gr.Blocks:
    """Create a Gradio app instance (for deployment)."""
    app = NetOpsDemo(model_path=model_path)
    app.load_model()
    return app.build_interface()


if __name__ == "__main__":
    app = NetOpsDemo()
    app.load_model()
    app.launch(server_name="0.0.0.0", server_port=7860, share=False)

from __future__ import annotations

import typer
from rich import print

from .model import NeuroSQLModel

app = typer.Typer(add_completion=False, help="NeuroSQL CLI: Text-to-SQL generation")


@app.command()
def main(
    query: str = typer.Option(..., "--query", "-q", help="Natural language question"),
    context: str = typer.Option(..., "--context", "-c", help="SQL schema context"),
    model: str = typer.Option("0xV4h3/neurosql", "--model", "-m", help="HF repo id or local path"),
    local_fallback: bool = typer.Option(True, "--local-fallback/--no-local-fallback", help="Use local weights fallback"),
    max_new_tokens: int = typer.Option(128, "--max-new-tokens"),
    num_beams: int = typer.Option(4, "--num-beams"),
    do_sample: bool = typer.Option(False, "--do-sample"),
    temperature: float = typer.Option(1.0, "--temperature"),
    top_p: float = typer.Option(1.0, "--top-p"),
):
    """
    Example:
    neuroscql --query "Show sales in Armenia for 2020" --context "CREATE TABLE sales(...);"
    """
    if local_fallback and model == "0xV4h3/neurosql":
        m = NeuroSQLModel.from_local_weights_or_hub(
            local_dir="neurosql/model_weights",
            hub_repo_id="0xV4h3/neurosql",
        )
    else:
        m = NeuroSQLModel.from_pretrained(model)

    sql = m.generate(
        query=query,
        context=context,
        max_new_tokens=max_new_tokens,
        num_beams=num_beams,
        do_sample=do_sample,
        temperature=temperature,
        top_p=top_p,
    )
    print(sql)


if __name__ == "__main__":
    app()
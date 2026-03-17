from __future__ import annotations

import typer
import uvicorn

app = typer.Typer(add_completion=False, help="Run NeuroSQL FastAPI server")


@app.command()
def main(
    host: str = typer.Option("0.0.0.0", "--host"),
    port: int = typer.Option(8000, "--port"),
    reload: bool = typer.Option(False, "--reload"),
):
    uvicorn.run("neurosql.api:app", host=host, port=port, reload=reload)


if __name__ == "__main__":
    app()
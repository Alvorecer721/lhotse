import click

from lhotse.bin.modes import prepare
from lhotse.recipes.thorsten_de import prepare_thorsten_de
from lhotse.utils import Pathlike

__all__ = ["thorsten_de"]


@prepare.command(context_settings=dict(show_default=True))
@click.argument("corpus_dir", type=click.Path(exists=True, dir_okay=True))
@click.argument("output_dir", type=click.Path())
def thorsten_de(
    corpus_dir: Pathlike,
    output_dir: Pathlike,
):
    """Thorsten-DE German speech data preparation."""
    prepare_thorsten_de(corpus_dir, output_dir=output_dir)

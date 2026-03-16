import click

from lhotse.bin.modes import prepare
from lhotse.recipes.nb_tale import prepare_nb_tale
from lhotse.utils import Pathlike

__all__ = ["nb_tale"]


@prepare.command(context_settings=dict(show_default=True))
@click.argument("corpus_dir", type=click.Path(exists=True, dir_okay=True))
@click.argument("output_dir", type=click.Path())
def nb_tale(
    corpus_dir: Pathlike,
    output_dir: Pathlike,
):
    """NB-Tale Norwegian speech data preparation."""
    prepare_nb_tale(corpus_dir, output_dir=output_dir)

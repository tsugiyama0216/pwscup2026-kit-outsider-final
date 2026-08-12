"""Run a tiny end-to-end OpenDP AIM synthesis without reading competition data."""

from __future__ import annotations

from importlib.metadata import version

import opendp.prelude as dp
import polars as pl


def make_toy_data() -> pl.DataFrame:
    """Return a small categorical table with a population-level dependency."""

    age_band = [0, 0, 1, 1, 2, 2] * 10
    sex = [0, 1, 0, 1, 0, 1] * 10
    outcome = [
        int((age >= 1 and idx % 3 == 0) or (age == 2 and idx % 2 == 0))
        for idx, age in enumerate(age_band)
    ]
    return pl.DataFrame(
        {
            "age_band": age_band,
            "sex": sex,
            "outcome": outcome,
        }
    )


def main() -> None:
    dp.enable_features("contrib")

    source = make_toy_data()
    rho = 0.1

    context = dp.Context.compositor(
        data=source.lazy(),
        privacy_unit=dp.unit_of(contributions=1),
        privacy_loss=dp.loss_of(rho=rho),
    )

    table = (
        context.query(rho=rho)
        .select("age_band", "sex", "outcome")
        .contingency_table(
            keys={
                "age_band": [0, 1, 2],
                "sex": [0, 1],
                "outcome": [0, 1],
            },
            algorithm=dp.mbi.AIM(queries=2, rounds=4, max_size=16.0),
        )
        .release()
    )

    synthetic = table.synthesize(rows=source.height, method="sample")

    expected_columns = {"age_band", "sex", "outcome"}
    if set(synthetic.columns) != expected_columns:
        raise AssertionError(f"unexpected columns: {synthetic.columns}")
    if synthetic.height != source.height:
        raise AssertionError(
            f"unexpected row count: {synthetic.height} (expected {source.height})"
        )

    print(f"OpenDP: {version('opendp')}")
    print(f"Private-PGM/mbi: {version('mbi')}")
    print(f"pandas: {version('pandas')}")
    print(f"Source shape: {source.shape}")
    print(f"Synthetic shape: {synthetic.shape}")
    print("AIM smoke test: OK")


if __name__ == "__main__":
    main()

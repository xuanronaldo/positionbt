import polars as pl


def load_close_data() -> pl.DataFrame:
    """Read test data from parquet files

    Returns:
        close_df: DataFrame with time and close price columns

    """
    # Read raw data from parquet file
    df = pl.read_parquet("data/btc_close_15m.parquet")

    # Process close price data
    close_df = df.select(pl.col("time"), pl.col("close"))

    return close_df

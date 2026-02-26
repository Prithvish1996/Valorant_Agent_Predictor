from IPython.display import display, HTML
import pandas as pd
def display_side_by_side(*dfs, titles=None, rows=5, index=False, gap="30px"):
    html_blocks = []
    for i, df in enumerate(dfs):
        title_html = ""
        if titles and i < len(titles):
            title_html = f"<h4 style='margin-bottom:8px'>{titles[i]}</h4>"

        table_html = df.head(rows).to_html(index=index)
        block = f"""
        <div>
            {title_html}
            {table_html}
        </div>
        """
        html_blocks.append(block)
    html = f"""
    <div style="display:flex; gap:{gap}; align-items:flex-start;">
        {''.join(html_blocks)}
    </div>
    """

    display(HTML(html))
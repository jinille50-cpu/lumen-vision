"""클래스 색 목록. Segmentation 예측 화면과 Labeler가 같은 색을 쓰도록 여기에 둔다.

클래스 0(배경)은 흰색, 나머지는 tab10 순서. 배경 빼고 10개를 넘으면 tab20을 쓴다.
"""

BACKGROUND = "#ffffff"
TAB10 = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
]
TAB20 = [
    "#1f77b4", "#aec7e8", "#ff7f0e", "#ffbb78", "#2ca02c",
    "#98df8a", "#d62728", "#ff9896", "#9467bd", "#c5b0d5",
    "#8c564b", "#c49c94", "#e377c2", "#f7b6d2", "#7f7f7f",
    "#c7c7c7", "#bcbd22", "#dbdb8d", "#17becf", "#9edae5",
]


def class_colors(num_classes):
    """클래스 번호 순서대로 '#rrggbb' 목록을 돌려준다 (0번은 흰색)"""
    n_fg = max(num_classes - 1, 0)          # 배경을 뺀 클래스 수
    table = TAB10 if n_fg <= len(TAB10) else TAB20
    # 20개를 넘으면 처음부터 다시 돈다 (% 는 나머지 연산)
    return [BACKGROUND] + [table[i % len(table)] for i in range(n_fg)]

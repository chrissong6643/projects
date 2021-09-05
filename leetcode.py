def pushDominoes(dominoes: str) -> str:
    if "L" not in dominoes and "R" not in dominoes:
        return dominoes
    if len(dominoes) == 1:
        return dominoes
    dominoes = list(dominoes)
    dir = []
    for i in range(len(dominoes)):
        if dominoes[i] == "L" or dominoes[i] == "R":
            dir.append([dominoes[i], i])
    i = 0
    while i in range(len(dir) - 1):
        if i < len(dir) - 1 and dir[i + 1][0] == "L" and dir[i][0] == "R":
            n = dir[i + 1][1] - dir[i][1] + 1
            if n % 2 == 0:
                s = list(("R" * (n // 2)) + ("L" * (n // 2)))
                dominoes[dir[i][1]:dir[i + 1][1] + 1] = s
            else:
                s = list(("R" * (n // 2)) + "." + ("L" * (n // 2)))
                dominoes[dir[i][1]:dir[i + 1][1] + 1] = s
        elif dir[i][0] == "L" and dir[i + 1][0] == "L":
            s = list("L" * (dir[i + 1][1] - dir[i][1] + 1))
            dominoes[dir[i][1]:dir[i + 1][1] + 1] = s
        elif dir[i][0] == "R" and dir[i + 1][0] == "R":
            s = list("R" * (dir[i + 1][1] - dir[i][1] + 1))
            dominoes[dir[i][1]:dir[i + 1][1] + 1] = s
        i += 1
    if dir[0][0] == "L":
        dominoes = list("L" * (dir[0][1] + 1)) + dominoes[dir[0][1] + 1:]
    if dir[-1][0] == "R":
        dominoes = dominoes[:dir[-1][1]] + list("R" * (len(dominoes) - dir[-1][1]))
    return "".join(dominoes)
















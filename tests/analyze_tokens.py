tokens = {}
with open('c4_dictionaries/bpe_superchunk/token_mapping', "r", encoding="utf-8") as file:
    for line in file:
        token = line[len(str(line.split()[0])) + 1:-1]
        if len(token) > 0:
            lower_token = token.lower()
            if lower_token not in tokens:
                tokens[lower_token] = set()
            tokens[lower_token].add(token)

first_cap_count = 0
for lower in tokens.keys():
    if len(tokens[lower]) > 1:
        var_list = list(tokens[lower])
        if len(var_list) == 2 and var_list[0][1:] == var_list[1][1:]:
            first_cap_count += 1
        else:
            print(f'{lower} ({len(tokens[lower])})-', end=' ')
            for token in list(tokens[lower]):
                print(f' {token} ', end=' ')
            print()

print('Only first letter capitalized: ', first_cap_count)
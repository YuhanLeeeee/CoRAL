import re


def format_prediction_output(raw_output: str):
    step_pattern = r'"(Elementary Step \d+)"'
    reactants_pattern = r'"reactants":\s*"(.*?)"'
    products_pattern = r'"products":\s*"(.*?)"'

    step_match = re.search(step_pattern, raw_output)
    react_match = re.search(reactants_pattern, raw_output)
    prod_match = re.search(products_pattern, raw_output)

    step_name = step_match.group(1) if step_match else "Inference Result"
    reactants = react_match.group(1) if react_match else "N/A"
    products = prod_match.group(1) if prod_match else "N/A"

    class Colors:
        HEADER = '\033[95m'
        BLUE = '\033[94m'
        GREEN = '\033[92m'
        BOLD = '\033[1m'
        RESET = '\033[0m'
        CYAN = '\033[96m'

    width = 80
    border = "=" * width
    divider = "-" * width

    print("\n" + border)

    title = f"✨ {step_name}"
    padding = (width - len(title)) // 2
    print(f"{Colors.BOLD}{Colors.HEADER}{' ' * padding}{title}{Colors.RESET}")
    print(border)

    print(f"\n{Colors.BLUE}{Colors.BOLD}[INPUT REACTANTS]:{Colors.RESET}")
    print(f"{Colors.CYAN}{reactants}{Colors.RESET}")

    print("\n" + divider)

    print(f"\n{Colors.GREEN}{Colors.BOLD}[PREDICTED PRODUCTS]:{Colors.RESET}")
    print(f"{Colors.GREEN}{products}{Colors.RESET}")

    print(border + "\n")


def parse_mechanism_output(raw_text):
    class Style:
        HEADER = '\033[95m'
        BLUE = '\033[94m'
        CYAN = '\033[96m'
        GREEN = '\033[92m'
        YELLOW = '\033[93m'
        BOLD = '\033[1m'
        RESET = '\033[0m'
        DIM = '\033[2m'

    final_pattern = r'"final_product":\s*"(.*?)"'
    final_match = re.search(final_pattern, raw_text)
    final_product = final_match.group(1) if final_match else "N/A"

    step_block_pattern = re.compile(r'"(Elementary Step \d+)":\s*\{(.*?)\}', re.DOTALL)
    steps = step_block_pattern.findall(raw_text)

    width = 80
    print("\n" + Style.BOLD + Style.HEADER + "╔" + "═" * (width - 2) + "╗" + Style.RESET)
    title = "⚛️  REACTION MECHANISM ANALYSIS REPORT"
    print(f"{Style.BOLD}{Style.HEADER}║{title:^{width - 2}}║{Style.RESET}")
    print(Style.BOLD + Style.HEADER + "╚" + "═" * (width - 2) + "╝" + Style.RESET + "\n")

    if steps:
        print(f"{Style.DIM}Found {len(steps)} elementary steps in reasoning process...{Style.RESET}\n")

        for i, (step_name, content) in enumerate(steps):
            r_match = re.search(r'"reactants":\s*"(.*?)"', content)
            p_match = re.search(r'"products":\s*"(.*?)"', content)

            r_val = r_match.group(1) if r_match else "Unknown"
            p_val = p_match.group(1) if p_match else "Unknown"

            print(f"{Style.YELLOW}{Style.BOLD}▶ {step_name}{Style.RESET}")
            print(f"  {Style.BOLD}Input :{Style.RESET} {Style.BLUE}{r_val}{Style.RESET}")
            print(f"  {Style.BOLD}Output:{Style.RESET} {Style.CYAN}{p_val}{Style.RESET}")

            if i < len(steps) - 1:
                print(f"      {Style.DIM}↓ (proceeds to next step){Style.RESET}")
                print(f"{Style.DIM}{'-' * 40}{Style.RESET}")
            else:
                print("\n")
    else:
        print(f"{Style.YELLOW}[!] No detailed mechanism steps detected inside <think> tags.{Style.RESET}\n")

    print(Style.BOLD + Style.GREEN + "╔" + "═" * (width - 2) + "╗" + Style.RESET)
    res_title = "✨ FINAL PREDICTION RESULT"
    print(f"{Style.BOLD}{Style.GREEN}║{res_title:^{width - 2}}║{Style.RESET}")
    print(Style.BOLD + Style.GREEN + "╠" + "═" * (width - 2) + "╣" + Style.RESET)

    print(f"{Style.BOLD}SMILES:{Style.RESET} {Style.GREEN}{final_product}{Style.RESET}")

    print(Style.BOLD + Style.GREEN + "╚" + "═" * (width - 2) + "╝" + Style.RESET + "\n")

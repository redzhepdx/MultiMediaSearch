import json

import requests


def retrieve_from_api(url: str, file_name: str = "data.json") -> None:
    """
    Retrieve data from API and save it to a file.
    :param url:
    :param file_name:
    """

    response = requests.get(url=url)
    data = response.json()
    with open(file_name, "w") as f:
        json.dump(data, f)


def main():
    url = "https://api.ggator.net/v1/ai/products"
    retrieve_from_api(url=url)


if __name__ == "__main__":
    main()

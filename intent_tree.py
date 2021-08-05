from __future__ import annotations
import aiohttp
import asyncio
import json
from collections import defaultdict
from schematics.models import Model
from schematics.types import BooleanType, StringType
import sys
from typing import List
from urllib.parse import urlencode
from urllib.request import urlopen


def asyncCached(f):
    cache = {}
    async def cached_f(*args, **kwargs):
        cache_key = (args, tuple(sorted(kwargs.items())))
        if cache_key in cache:
            return cache[cache_key]
        else:
            result = await f(*args, **kwargs)
            cache[cache_key] = result
            return result

    cached_f.__name__ = f.__name__
    return cached_f


class RequestError(Exception):
    def __init__(self, message):
        super().__init__(message)


@asyncCached
async def requestIntent(phrase: str) -> str:
    async with aiohttp.ClientSession() as session:
        url_template = 'https://sandbox.twin24.ai/parse'
        params = urlencode({'q': phrase})
        url = '{url_template}?{params}'.format(**locals())
        async with session.get(url) as response:
            if response.status != 200:
                raise RequestError(
                    'ERROR {response.status} GET {url}'.format(**locals()),
                )

            data = await response.json()
            intent = data.get('intent', {}).get('name')
            print('{phrase} => {intent}'.format(**locals()))
            return intent


class DialogPhrase(Model):
    text = StringType(required=True)
    is_bot = BooleanType(required=True)
    intent = StringType()


Dialog = List[DialogPhrase]


def readDialogFromFile(fileName: str) -> Dialog:
    with open(fileName, encoding='utf8') as f:
        return [DialogPhrase(item) for item in json.loads(f.read())]


class IntentTreeNode:
    def __init__(self, intent: str):
        self.intent = intent
        self.phrases = set()
        self.childrenMap = {}

    def lookupChild(self, intent: str):
        return self.childrenMap.get(intent)

    def addChild(self, node: 'IntentTreeNode'):
        self.childrenMap[node.intent] = node

    def registerPhrase(self, text: str):
        self.phrases.add(text)

    def children(self):
        for _, child in self.childrenMap.items():
            yield child


class IntentTreeBuilder:
    ROOT_INTENT = 'ROOT_INTENT'
    BOT_INTENT = 'DETERMINISTIC_BOT_INTENT'

    def __init__(self):
        self.treeRoot = IntentTreeNode(self.ROOT_INTENT)

    async def addDialog(self, dialog: List[DialogPhrase]):
        node = self.treeRoot
        for phrase in dialog:
            if phrase.is_bot:
                intent = self.BOT_INTENT
            else:
                intent = await requestIntent(phrase.text)

            childNode = node.lookupChild(intent)
            if childNode is None:
                childNode = IntentTreeNode(intent)
                node.addChild(childNode)
            childNode.registerPhrase(phrase.text)

            node = childNode

    def intentTree(self) -> List(dict):
        return IntentTreeBuilder._intentSubtree(self.treeRoot)

    @classmethod
    def _intentSubtree(cls, node: IntentTreeNode) -> List(dict):
        data = []
        for child in node.children():
            is_bot = child.intent == cls.BOT_INTENT
            nodeData = {
                'is_bot': is_bot,
                'phrases': sorted(child.phrases),
                'replies': cls._intentSubtree(child),
            }
            if not is_bot:
                nodeData['intent'] = child.intent
            data.append(nodeData)
        return data


async def buildIntentTree(dialogs: List[Dialog]) -> List(dict):
    builder = IntentTreeBuilder()
    await asyncio.gather(
        *[
            builder.addDialog(dialog)
            for dialog in dialogs
        ],
    )
    return builder.intentTree()


if __name__ == "__main__":
    dialogs = [
        readDialogFromFile(fileName)
        for fileName in sys.argv[1:]
    ]
    print(
        json.dumps(
            asyncio.run(
                buildIntentTree(dialogs),
            ),
            ensure_ascii=False,
            indent=4,
            sort_keys=True,
        ),
    )

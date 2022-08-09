#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from fastapi_app import Data, predict
import asyncio

if __name__ == "__main__":
    data = Data(
            question="Where did Neil Armstrong study?",
            backend="Triton Inference Server"
    )
    output = asyncio.run(predict(data))
    print(output)



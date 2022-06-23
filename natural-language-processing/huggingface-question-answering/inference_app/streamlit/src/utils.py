#!/usr/bin/env python3
# -*- coding: utf-8 -*-

css = """
<style>
.userTextRoot {
        display: flex;
        height: auto;
        margin: 0px;
        width: 100%;
}

.botTextRoot {
    display: flex;
    height: auto;
    margin: 0px;
    width: 100%;
    flex-direction: row-reverse;
}

.chatPic {
    border: 1px solid transparent;
    border-radius: 50%;
    height: 3rem;
    width: 3rem;
    margin: 0px;
}

.chatText {
    display: inline-block;
    background: rgb(240, 242, 246);
    border: 1px solid transparent;
    border-radius: 10px;
    padding: 10px 14px;
    margin: 5px 20px;
    max-width: 70%;
}
</style>
"""


user_component = """
<div class="userTextRoot">
    <img src="https://img.icons8.com/color/48/undefined/user.png" draggable="false" class="chatPic"/>
    <div class="chatText">
        %s
    </div>
</div>
"""

bot_component = """
<div class="botTextRoot">
    <img src="https://img.icons8.com/fluency/48/undefined/chatbot.png" draggable="false" class="chatPic"/>
    <div class="chatText">
        %s
    </div>
</div>
"""


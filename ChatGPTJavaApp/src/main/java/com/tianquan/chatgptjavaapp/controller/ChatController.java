package com.tianquan.chatgptjavaapp.controller;

import com.tianquan.chatgptjavaapp.service.ChatService;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.MediaType;
import org.springframework.stereotype.Controller;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestMethod;
import org.springframework.web.bind.annotation.ResponseBody;

import java.io.IOException;
import java.nio.charset.StandardCharsets;

@Slf4j
@Controller
public class ChatController {

    @Autowired
    private ChatService chatService;

    @RequestMapping(value = "/chat/{content}", method = RequestMethod.GET, produces = MediaType.APPLICATION_JSON_VALUE)
    @ResponseBody
    public String getResponses(@PathVariable(value="content") String content) throws IOException {
//        System.out.println(content);
        String decoded = java.net.URLDecoder.decode(content, StandardCharsets.UTF_8);
        System.out.println("Current input chat is: " + decoded);
        String chatResponse = chatService.getChatResponse(decoded);
        return chatResponse;
    }

    @RequestMapping("/test")
    @ResponseBody
    public String dummyMessage() {
        return "hello world";
    }
}

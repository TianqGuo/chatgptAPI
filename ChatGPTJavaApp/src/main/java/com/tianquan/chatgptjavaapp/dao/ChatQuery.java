package com.tianquan.chatgptjavaapp.dao;

import com.tianquan.chatgptjavaapp.utils.RequestsGenerator;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Repository;

import java.io.IOException;


@Repository
public class ChatQuery {

    @Autowired
    RequestsGenerator generator;

    public String queryChat(String content) throws IOException {
        String response = generator.generateChat(content);
        System.out.println("ChatQuery: " + response);
        return response;
    }
}

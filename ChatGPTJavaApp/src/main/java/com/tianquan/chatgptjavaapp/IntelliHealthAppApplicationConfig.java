package com.tianquan.chatgptjavaapp;

import com.tianquan.chatgptjavaapp.utils.RequestsGenerator;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.web.servlet.config.annotation.EnableWebMvc;

@Configuration
@EnableWebMvc
public class IntelliHealthAppApplicationConfig {
    @Bean(name = "generator")
    public RequestsGenerator newGenerator() {
        RequestsGenerator generator = new RequestsGenerator();
        return generator;
    }
}

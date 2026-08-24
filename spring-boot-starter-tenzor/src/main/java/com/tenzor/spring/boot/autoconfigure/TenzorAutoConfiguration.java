package com.tenzor.spring.boot.autoconfigure;

import com.tenzor.spring.boot.TenzorTemplate;
import org.springframework.boot.autoconfigure.condition.ConditionalOnClass;
import org.springframework.boot.autoconfigure.condition.ConditionalOnMissingBean;
import org.springframework.boot.context.properties.EnableConfigurationProperties;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

@Configuration
@ConditionalOnClass(name = "com.tenzor.core.Tensor")
@EnableConfigurationProperties(TenzorProperties.class)
public class TenzorAutoConfiguration {

    @Bean
    @ConditionalOnMissingBean
    public TenzorTemplate tenzorTemplate(TenzorProperties properties) {
        return new TenzorTemplate(properties);
    }
}

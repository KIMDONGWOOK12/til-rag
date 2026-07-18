배운 내용 : Transformer의 마스킹, 잔차 연결, 레이어 정규화

**마스킹(Masking)**

*   어텐션 연산에서 특정 위치의 정보를 참조하지 못하도록 차단하는 기법이다.
    
    *   사용 이유 : 패딩 토큰을 무시하거나 디코더에서 미래 토큰을 미리 보지 못하게 하기 위해서
        
    *   종류
        
        *   패딩 마스크: PAD 토큰 위치 차단 (인코더, 디코더 모두 사용)
            
        *   룩어헤드 마스크: 디코더에서 현재 위치 이후 토큰 차단 (개발자가 상삼각 행렬로 지정)
            
            *   인과적 마스킹: 학습 중 미래 토큰 미리보기 금지
                
    *   동작 방식
        
        *   차단할 위치에 -oo 대입 -> Softmax 후 가중치가 0이 됨 -> 해당 위치 정보 완전 차단
            

    # 룩어헤드 마스크 생성
    look_ahead_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
    scores = scores.masked_fill(look_ahead_mask.bool(), float("-inf"))
    weights = F.softmax(scores, dim=-1)

**잔차 연결 (Residual Connection)**

*   레이어의 입력을 출력을 직접 더해 변화량(잔차)만 학습하도록 만드는 기법
    
    *   사용 이유
        
        *   깊은 신경망에서 기울기 소실 문제를 줄이고 학습을 안정화하기 위해
            
        *   핵심 수식
            

    output = x + Sublayer(x)
    
    역전파: d/dx(MHA(x) + x) = d/dx MHA(x) + 1
    -> 기울기에 항상 1이 더해져 소실 방지

*   코드에서 보면
    

    x = self.norm1(x + attn_out)
    x = self.norm2(x + ffn_out)

**레이어 정규화 (Layer Normalization)**

각 샘플의 특성 차원을 기준으로 평균과 분산을 계산해 정규화하는 기법이다.

*   사용 이유
    
    *   레이어 출력 분포가 학습 중 흔들리는 것을 막아 학습 속도와 안정성을 높이기 위해
        
*   수식
    

![file_iwXNvsgNhhjbNptpgw](https://exp-upload.goorm.io/2026-06-11/á/iwXNvsgNhhjbNptpgwwebp)

*   Transformer에서 사용 위치
    

    x = LayerNorm(x + Sublayer(x)) # 잔차 연결 후 직후 적용

    layer_norm = nn.LayerNorm(d_model)
    x = layer_norm(x + sublayer_output)

**위클리 챌린지 2번: VGG16 + Fine-Tuning 핵심**

1\. VGG16 불러오기 (사전훈련 가중치)

2\. 전체 레이어 동결

3\. 마지막 분류 레이어만 교체

4\. 마지막 레이어만 학습

5\. 상위 레이어 동결 해제 + 재학습

6\. 평가

핵심 코드 ( 개인 소견 )

    # 전체 동결
    for param in model_ft.parameters():
        param.requires_grad = False
    
    model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes) # 마지막 레이어만 교체
    
    optimizer = optim.Adam(model_ft.classifier[6].parameters(), lr=0.001) # 마지막 레이어만 학습

**Fine-tuning 포인트**

*   하위 레이어: 동결 유지 (선, 색상 등 기본 특징, 공통 사용)
    

*   상위 레이어: 동결 해제 (새 데이터에 특화된 특징 학습 필요)
    
*   전체 해제하면 과적합 위험 (VGG16 파라미터 1억 3800만개)
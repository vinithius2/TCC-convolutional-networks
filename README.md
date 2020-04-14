
# tcc-redes-convolucionais

    1. CONTEXTUALIZAÇÃO

Com uma grande demanda e uma administração sem nenhuma ferramenta adequada de controle de fluxo, as UPAS trazem um atendimento inadequado, muitas horas de espera, alta demanda e poucos médicos, que por consequência, tratam toda a população com total descaso, mães com seus filhos que não conseguem um rápido atendimento com um pediatra, ou um idoso hipertenso que não consegue ser atendido com prioridade. 	
Os estabelecimentos da UPA que trabalham com atendimento médico 24 horas por dia, precisariam de um dado estatístico para ter uma previsão de quantidade e categoria do público, para controlar melhor o fluxo e solicitar médicos específicos para os dias em que a demanda de uma categoria for maior, por exemplo, se a demanda de atendimento para crianças em um dia especifico for comprovado estatisticamente, o atendimento médico especializado já estaria de prontidão.


    2. PROJETO

Este projeto foi elaborado pensando em resolver tais problemas contextualizados acima, ele é capas de verificar em tempo real pela WebCam cada rosto e válidar categoricamente se é um **homem adulto** *(431)*, **homem jovem** *(171)*, **homem velho** *(204)*, **mulher adulta** *(452)*, **mulher jovem** *(181)* e **mulher velha** *(185)*, o projeto utilizou um **dataset** com **1624** imagens com canal cinza de **48 x 48 px** para o treinamento da inteligência artificial. 

Para que ocorra o registro é necessário ter um rosto humano com mais de 50% de probabilidade de ser alguma categoria citada acima, quando cadastrado é salvo em memoria o **padrão característico facial* até o fim da execução, pois a cada face é validada para saber se já foi cadastrada na execução, caso já tenha o registro do padrão característico facial, não salva novamente a mesma pessoa, se não, salva respeitando a probabilidade de ser acima de 50% da caraterística encontrada.

Após finalizado a execução, é salvo em um arquivo CSV as colunas **faces**, **categoria**, **probabilidade**, **data**, **hora**, ex: 

| faces | categoria | probabilidade | data | hora |

| 1 | adult_male | 99.43 | 12/04/2020 | 16:51:38 |

| 2 | adult_male | 84.29 | 12/04/2020 | 16:53:13 |

É possível gerar gráficos os arquivos CSV, Ex:
*(Valores fictícios)*

![Bar](https://user-images.githubusercontent.com/7644485/79261410-2920c980-7e66-11ea-8089-ef5b709f7132.png)
![Line](https://user-images.githubusercontent.com/7644485/79261431-2f16aa80-7e66-11ea-8cfc-6a707c7f16f8.png)
![Pie](https://user-images.githubusercontent.com/7644485/79261440-3473f500-7e66-11ea-8e9b-166fe47126d0.png)

    3. COMANDOS

 - Descompacta as imagens e cria o *dataset* para rede neural.
	 - main.py -d --dataset
	 
 - Treina e testar a rede neural, gerando gráficos para o entendimento
   do treinamento.
	 - main.py -t --training
	 
 - Inicia a detecção na imagem passada pelo PATH
	 - main.py -i <path> --image <path>

 - Inicia a detecção no vídeo passado pelo PATH
	 - main.py -v <path> --video <path>

 - Inicia a detecção em tempo real pela webcam, o uso do "--save" é
   opcional caso seja chamado, salvando assim o vídeo atual.
	 - main.py -r --real --save

 - Gerar gráficos estatísticos dos arquivos CSV que foram gerados e se
   encontram no diretório do projeto "material/csv_data/", Ex: **main.py
   -s 01.csv -t pie** *Tipos:* **pie, line, bar**
	 - main.py -s <path> --statistics <path> -t <type> --type <type>

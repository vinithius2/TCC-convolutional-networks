
# tcc-redes-convolucionais (tcc-convolutional-networks)

Link do TCC _(TCC Link)_: https://drive.google.com/file/d/1b8DBn_dpyp5Y5mFKk7Ucv8R65pr8JmiT/view?usp=sharing

    1. CONTEXTUALIZAÇÃO (CONTEXTUALIZATION)

_**PT:**_ Com uma grande demanda e uma administração sem nenhuma ferramenta adequada de controle de fluxo, as UPAS trazem um atendimento inadequado, muitas horas de espera, alta demanda e poucos médicos, que por consequência, tratam toda a população com total descaso, mães com seus filhos que não conseguem um rápido atendimento com um pediatra, ou um idoso hipertenso que não consegue ser atendido com prioridade. 	
Os estabelecimentos da UPA que trabalham com atendimento médico 24 horas por dia, precisariam de um dado estatístico para ter uma previsão de quantidade e categoria do público, para controlar melhor o fluxo e solicitar médicos específicos para os dias em que a demanda de uma categoria for maior, por exemplo, se a demanda de atendimento para crianças em um dia especifico for comprovado estatisticamente, o atendimento médico especializado já estaria de prontidão.

_**ENG:**_ With a high demand and an administration without any adequate flow control tool, the UPAS bring inadequate care, many hours of waiting, high demand and few doctors, who consequently treat the entire population with total disregard, mothers with their children who cannot get quick care from a pediatrician, or a hypertensive elderly person who cannot be given priority care.
UPA establishments that work with 24-hour medical care would need statistical data to forecast the number and category of the public, to better control the flow and request specific doctors for the days when the demand for a category is higher, for example, if the demand for care for children on a specific day is statistically proven, the specialized medical care would already be ready.

    2. PROJETO (PROJECT)

_**PT:**_ Este projeto foi elaborado pensando em resolver tais problemas contextualizados acima, ele é capas de verificar em tempo real pela WebCam cada rosto e válidar categoricamente se é um **homem adulto** *(431)*, **homem jovem** *(171)*, **homem velho** *(204)*, **mulher adulta** *(452)*, **mulher jovem** *(181)* e **mulher velha** *(185)*, o projeto utilizou um **dataset** com **1624** imagens com canal cinza de **48 x 48 px** para o treinamento da inteligência artificial. 

_**ENG:**_ This project was prepared thinking about solving such problems contextualized above, it is able to verify in real time by WebCam each face and categorically validate if it is a **adult man** *(431)*, **young man** *( 171)*, **old man** *(204)*, **adult woman** *(452)*, **young woman** *(181)* and **old woman** *(185) *, the project used a **dataset** with **1624** images with a gray channel of **48 x 48 px** for artificial intelligence training.

![gray](https://user-images.githubusercontent.com/7644485/79281334-e6bcb400-7e88-11ea-87c1-21d8d036f008.png)

_**PT:**_ Para que ocorra o registro é necessário ter um rosto humano com mais de 50% de probabilidade de ser alguma categoria citada acima, quando cadastrado é salvo em memoria o **padrão característico facial* até o fim da execução, pois a cada face é validada para saber se já foi cadastrada na execução, caso já tenha o registro do padrão característico facial, não salva novamente a mesma pessoa, se não, salva respeitando a probabilidade de ser acima de 50% da caraterística encontrada.

_**ENG:**_ For the registration to occur, it is necessary to have a human face with more than 50% probability of being any category mentioned above, when registered, the **facial characteristic pattern* is saved in memory until the end of the execution, as each face is validated to find out if it has already been registered in the execution, if it already has the registration of the facial characteristic pattern, it does not save the same person again, if not, it saves respecting the probability of being above 50% of the found characteristic.

![detect](https://user-images.githubusercontent.com/7644485/79281264-c1c84100-7e88-11ea-8289-bf0bd41d92b6.png)

_**PT:**_ Após finalizado a execução, é salvo em um arquivo CSV as colunas **faces**, **categoria**, **probabilidade**, **data**, **hora**, ex: 

_**ENG:**_ After the execution is finished, the columns **faces**, **category**, **probability**, **date**, **time**, eg:

| faces | categoria | probabilidade | data | hora |
--- | --- | --- | --- | ---
| 4 | young_female | 56.06 | 27/05/2020 | 20:25:44 |
| 4 | young_female | 55.14 | 27/05/2020 | 20:25:44 |

_**PT:**_ É possível gerar gráficos dos arquivos CSV processados, Ex:
*(Valores fictícios)*

_**ENG:**_ It is possible to generate graphics from the processed CSV files, eg:
*(Fictitious values)*

![Bar](https://user-images.githubusercontent.com/7644485/83085667-01c33c00-a063-11ea-8492-5ce4ab56448c.png)
![Line](https://user-images.githubusercontent.com/7644485/83085685-0b4ca400-a063-11ea-9346-6626eefcdb12.png)
![Pie](https://user-images.githubusercontent.com/7644485/83085710-130c4880-a063-11ea-8fad-34fba3dafcc5.png)

    3. COMANDOS (COMMANDS)

 - Descompacta as imagens e cria o *dataset* para rede neural. _(Unzip the images and create the *dataset* for the neural network.)_
	 - main.py -d --dataset
	 
 - Treina e testar a rede neural, gerando gráficos para o entendimento do treinamento. _(Train and test the neural network, generating graphics for training understanding.)_
	 - main.py -t --training
	 
 - Inicia a detecção na imagem passada pelo PATH _(Start detection on image passed by PATH)_
	 - main.py -i <path> --image <path>

 - Inicia a detecção no vídeo passado pelo PATH _(Start detection on video passed by PATH)_
	 - main.py -v <path> --video <path>

 - Inicia a detecção em tempo real pela webcam, o uso do "--save" é opcional caso seja chamado, salvando assim o vídeo atual. _(Start real-time detection by webcam, use of "--save" is optional if called, thus saving current video.)_
	 - main.py -r --real --save

 - Gerar gráficos estatísticos dos arquivos CSV que foram gerados e se encontram no diretório do projeto "material/csv_data/", Ex: **main.py -s 01.csv -t pie** *Tipos:* **pie, line, bar** _(Generate statistical graphics of the CSV files that were generated and are located in the project directory "material/csv_data/", Ex: main.py -s 01.csv -t pie Types: pie, line, Pub)_
	 - main.py -s <path> --statistics <path> -t <type> --type <type>

import numpy as np 

umbral = 0.5
tasa_de_aprendizaje = 0.1
pesos_iniciales = np.array([
	[0], 
	[0], 
	[0]
])#Pesos inicales sin entranar

conjunto_de_entrenamiento = np.array([
	[1, 0, 0], 
	[1, 0, 1],
	[1, 1, 0],
	[1, 1, 1]
])#Tabla compuesta NAND

print "Conjunto de entramiento: %s "%(conjunto_de_entrenamiento)
print "-"*60

salidas_esperadas = np.array([
	[1],
	[1],
	[1],
	[0]
])#Resultado esperados de la tabla NAND

print "Resultados esperados: %s"%(salidas_esperadas)

print "-"*60
print "Pesos inciales del perceptron: %s"%(pesos_iniciales)


def activacion(sum, umbral):
	if(sum > umbral): return 1 
	return 0

def calcular_error(salida, resultado_parcial):
	return salida-resultado_parcial

def ajustar_pesos(entradas, pesos, ajuste):
	suma_entradas_ajuste = np.array([entradas*ajuste]).T
	pesos = np.add(pesos, suma_entradas_ajuste)#Actualiza los pesos, sumandole el ajuste
	return pesos#Retorna los pesos actualizados 

def entrenamiento(entradas, salidas, pesos):
	print "-"*60
	print "Comienza entranamiento de la neurona."

	for entreno in range(0, 10000):
		#Entranar el perceptron 10000 veces
		for i in range(0, len(entradas)):		
			sum = np.dot(entradas[i], pesos)#Sumatoria de entradas y pesos
			parcial = activacion(sum, umbral)#Resultado parcial
			error = calcular_error(salidas[i], parcial)#Error cometido, (Aprendizaje supervisado)
			if(entreno%1000 == 0): print "Promedio del error cometido: %s"%(np.average(np.abs(error)))#Promedio del error cometido
			ajuste = tasa_de_aprendizaje*error#Ajustar el error en base a la tasa de aprendizaje 
			pesos = ajustar_pesos(entradas[i], pesos, ajuste[0])#Actualiza los nuevos valores de los pesos ajustados
	return pesos#retona los pesos ajustados
			
def prueba(entrada, pesos):
	sum = np.dot(entrada, pesos)
	print "Para este caso: %s  la respuesta deberia ser: %s"%(entrada, activacion(sum, umbral))   

pesos_entrenados = entrenamiento(conjunto_de_entrenamiento, salidas_esperadas ,pesos_iniciales)
print "-"*60
prueba([1, 1, 1], pesos_entrenados)


#Fuentes: https://es.wikipedia.org/wiki/Perceptr%C3%B3n#Ejemplo

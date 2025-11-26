import numpy as np
import math
from typing import Optional, Tuple, Iterable


class OccupancyGrid:
	"""Grid de ocupação simples em memória.

	- `grid` usa: -1 = desconhecido, 0 = livre, 1 = ocupado
	- `visits` conta quantas vezes uma célula foi observada
	"""

	def __init__(
		self,
		size: int = 200,
		resolution: float = 0.05,
		existing_grid: Optional[np.ndarray] = None,
		existing_visit: Optional[np.ndarray] = None,
	) -> None:
		self.size = int(size)
		self.res = float(resolution)

		if existing_grid is None:
			self.grid = -np.ones((self.size, self.size), dtype=int)
			self.visits = np.zeros((self.size, self.size), dtype=int)
		else:
			self.grid = existing_grid
			self.visits = (
				existing_visit
				if existing_visit is not None
				else np.zeros((self.size, self.size), dtype=int)
			)

	def world_to_grid(self, x: float, y: float) -> Tuple[int, int]:
		"""Converte coordenadas do mundo (metros) para índices de grade.

		Origem da grade está no centro (`size/2`).
		"""
		i = int(self.size / 2 + x / self.res)
		j = int(self.size / 2 + y / self.res)
		return i, j

	def grid_to_world(self, i: int, j: int) -> Tuple[float, float]:
		"""Converte índices da grade de volta para coordenadas do mundo (m)."""
		x = (i - self.size / 2) * self.res
		y = (j - self.size / 2) * self.res
		return x, y

	def update(
		self,
		pose: Tuple[float, float, float],
		angle: float,
		dist: float,
		max_dist: float = 2.0,
		steps: int = 20,
	) -> None:
		"""Atualiza a grade com uma leitura de sensor (ray cast simplificado).

		- `pose`: (px, py, yaw)
		- `angle`: ângulo relativo do sensor (radianos)
		- `dist`: distância medida (m). Se igual a `max_dist` assume 'sem hit'
		- `steps`: número de células marcadas como livres ao longo do raio
		"""
		px, py, yaw = pose

		# Proteções básicas
		if dist is None or dist < 0:
			return

		# Marcar células livres ao longo do raio (exclui o ponto final)
		for k in range(steps):
			d = (dist * k) / steps
			x = px + d * math.cos(yaw + angle)
			y = py + d * math.sin(yaw + angle)
			i, j = self.world_to_grid(x, y)
			if 0 <= i < self.size and 0 <= j < self.size:
				self.grid[i, j] = 0
				self.visits[i, j] += 1

		# Se detectou um obstáculo antes da distância máxima, marcar a célula final como ocupada
		if dist < max_dist:
			x = px + dist * math.cos(yaw + angle)
			y = py + dist * math.sin(yaw + angle)
			i, j = self.world_to_grid(x, y)
			if 0 <= i < self.size and 0 <= j < self.size:
				self.grid[i, j] = 1
				self.visits[i, j] += 1

	def as_array(self) -> np.ndarray:
		"""Retorna referência da matriz interna de ocupação."""
		return self.grid
